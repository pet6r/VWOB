import numpy as np
from vispy import app, scene
from vispy.visuals import SphereVisual
import websocket
import json
import time
import threading
import os
import base64
import hashlib
import hmac
import datetime

# API Connection to environment variables
API_KEY = os.getenv('apikey')
API_SECRET = os.getenv('secretkey')

# WebSocket message handler
current_price = 93600  # Example price, replace with dynamic logic
order_data = {"bids": [], "asks": []}  # Temporary storage for bids/asks

# Generate RFC 3339 timestamps
def generate_time_in():
    """Generates the start time (10 seconds before the current time) in RFC 3339 format."""
    start_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)
    return start_time.isoformat() + "Z"

def generate_time_out():
    """Generates the current time in RFC 3339 format."""
    end_time = datetime.datetime.utcnow()
    return end_time.isoformat() + "Z"

# Function to create a Kraken WebSocket signature
def get_signature(api_path, nonce, post_data, secret):
    message = (str(nonce) + post_data).encode()
    secret = base64.b64decode(secret)
    hashed = hmac.new(secret, message, hashlib.sha512)
    return base64.b64encode(hashed.digest()).decode()

def on_message(ws, message):
    global order_data, current_price
    print("Message received:", message)
    try:
        data = json.loads(message)
        if data.get("type") == "snapshot":
            # Process snapshot data
            for entry in data["data"]:
                order_data["bids"] = entry.get("bids", [])
                order_data["asks"] = entry.get("asks", [])
                for bid in order_data["bids"]:
                    plot_mountain(float(bid["price"]), float(bid["qty"]), time.time(), quadrant=1)
                for ask in order_data["asks"]:
                    plot_mountain(float(ask["price"]), float(ask["qty"]), time.time(), quadrant=2)
        elif data.get("type") == "update":
            # Process incremental updates
            for entry in data["data"]:
                for bid in entry.get("bids", []):
                    plot_mountain(float(bid["price"]), float(bid["qty"]), time.time(), quadrant=1)
                for ask in entry.get("asks", []):
                    plot_mountain(float(ask["price"]), float(ask["qty"]), time.time(), quadrant=2)
    except Exception as e:
        print("Error processing message:", e)


def on_open(ws):
    subscribe_data = {
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": "XBT/USD",
            "depth": 10,
            "snapshot": True
        },
        "success": True,
        "time_in": str(generate_time_in()),
        "time_out": str(generate_time_out())
    }
    ws.send(json.dumps(subscribe_data))
    print("Sent subscription request:", json.dumps(subscribe_data, indent=4))


# Connect to Kraken WebSocket API
url = "wss://ws.kraken.com/v2"
ws = websocket.WebSocketApp(url, on_message=on_message, on_open=on_open)

# Canvas with 3D scene
canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=True)
view = canvas.central_widget.add_view()

# Camera setup
view.camera = "turntable"
view.camera.distance = 8000
view.camera.center = (0, 0, 0)

# Axis and grid setup
def create_axis_line(start, end, color, parent):
    pos = np.array([start, end])
    axis = scene.visuals.Line(pos=pos, color=color, parent=parent, width=2)
    return axis


create_axis_line([-5000, 0, 0], [5000, 0, 0], (1, 0, 0, 1), view.scene)  # X-axis
create_axis_line([0, -5000, 0], [0, 5000, 0], (0, 1, 0, 1), view.scene)  # Y-axis
create_axis_line([0, 0, -5000], [0, 0, 5000], (0, 0, 1, 1), view.scene)  # Z-axis

# Function to plot mountains
def plot_mountain(price, volume, timestamp, quadrant):
    global current_price
    x_offset = price - current_price
    z_offset = -(timestamp % 5000)  # Simulate time progression

    if quadrant == 1:  # Bids
        x = abs(x_offset)
        y = volume
        z = z_offset
    elif quadrant == 2:  # Asks
        x = -abs(x_offset)
        y = volume
        z = z_offset

    vertex = np.array([[x, y, z]])
    mountain = scene.visuals.Markers()
    mountain.set_data(vertex, edge_color=None, face_color=(0.2, 0.8, 1.0, 1.0), size=5)
    mountain.parent = view.scene


# Timer for visualization updates
def update_visualization(event):
    # Clear and replot data every 10 seconds
    for item in order_data["bids"]:
        plot_mountain(float(item["price"]), float(item["qty"]), time.time(), quadrant=1)
    for item in order_data["asks"]:
        plot_mountain(float(item["price"]), float(item["qty"]), time.time(), quadrant=2)


timer = app.Timer(interval=10, connect=update_visualization, start=True)

# Run WebSocket in a separate thread
websocket_thread = threading.Thread(target=ws.run_forever)
websocket_thread.daemon = True
websocket_thread.start()

# Run the application
app.run()
