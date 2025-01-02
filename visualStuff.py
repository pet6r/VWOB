import numpy as np
from vispy import app, scene
from vispy.visuals import SphereVisual
from vispy.visuals.transforms import MatrixTransform
import websocket
import json
import time
import base64
import hashlib
import hmac
import os
import threading

# Initialize global variables
bid_volumes = np.zeros(10)
ask_volumes = np.zeros(10)

# API Connection to environment variables
API_KEY = os.getenv('apikey')
API_SECRET = os.getenv('secretkey')

# Function to create a Kraken WebSocket signature
def get_signature(api_path, nonce, post_data, secret):
    message = (str(nonce) + post_data).encode()
    secret = base64.b64decode(secret)
    hashed = hmac.new(secret, message, hashlib.sha512)
    return base64.b64encode(hashed.digest()).decode()

# Ensure bid_volumes and ask_volumes are initialized
bid_volumes = np.zeros(10)
ask_volumes = np.zeros(10)

# Real-time rendering update with fallback for scaling
# Real-time rendering update
def update(event):
    global bid_volumes, ask_volumes

    # Check if volumes are available
    if len(bid_volumes) > 0 and len(ask_volumes) > 0:
        # Interpolate to match the vertex count
        ask_volumes_interp = np.interp(np.linspace(0, len(ask_volumes) - 1, asks_mountain_vertices.shape[0]), np.arange(len(ask_volumes)), ask_volumes)
        bid_volumes_interp = np.interp(np.linspace(0, len(bid_volumes) - 1, bids_mountain_vertices.shape[0]), np.arange(len(bid_volumes)), bid_volumes)

        # Update mountains based on new volumes
        asks_mountain_vertices[:, 1] = 600 * ask_volumes_interp / np.max(ask_volumes_interp)
        bids_mountain_vertices[:, 1] = 600 * bid_volumes_interp / np.max(bid_volumes_interp)

        # Dynamically change colors based on volume magnitude
        ask_color_intensity = np.clip(ask_volumes_interp / np.max(ask_volumes_interp), 0.1, 1.0)
        bid_color_intensity = np.clip(bid_volumes_interp / np.max(bid_volumes_interp), 0.1, 1.0)

        asks_mountain.mesh_data.set_vertices(asks_mountain_vertices)
        bids_mountain.mesh_data.set_vertices(bids_mountain_vertices)

        asks_mountain.color = (0.2, 0.2, ask_color_intensity.mean(), 1.0)
        bids_mountain.color = (0.2, 0.2, bid_color_intensity.mean(), 1.0)

# Add a safeguard to ensure WebSocket updates are properly received
def on_message(ws, message):
    global bid_volumes, ask_volumes
    print(f"Message received: {message}")  # Debugging WebSocket messages
    try:
        data = json.loads(message)
        if isinstance(data, list) and len(data) > 1:
            if "b" in data[1]:  # Handle bid data
                bids = data[1]["b"]
                if bids:  # Ensure there are valid bids
                    bid_volumes = np.array([float(b[1]) for b in bids])
            if "a" in data[1]:  # Handle ask data
                asks = data[1]["a"]
                if asks:  # Ensure there are valid asks
                    ask_volumes = np.array([float(a[1]) for a in asks])
    except Exception as e:
        print(f"Error parsing message: {e}")


# WebSocket connection open handler
def on_open(ws):
    # Define subscription message for public order book
    subscribe_data = {
        "event": "subscribe",
        "pair": ["XBT/USD"],
        "subscription": {"name": "book", "depth": 10}
    }

    # Send subscription message
    ws.send(json.dumps(subscribe_data))

# Connect to Kraken WebSocket API
url = "wss://ws.kraken.com/"
ws = websocket.WebSocketApp(url, on_message=on_message, on_open=on_open)

# Visualization Setup
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', show=True)
view = canvas.central_widget.add_view()

# Camera Settings
view.camera = 'turntable'
view.camera.distance = 6700
view.camera.center = (0, 0, 3000)
view.camera.elevation = -5
view.camera.azimuth = 0
view.camera.fov = 60

# Background and Sun
background_data = np.linspace([0, 0, 0], [0.5, 0, 0.5], 500).astype('float32')
background = scene.visuals.Image(background_data.reshape(1, 500, 3), parent=canvas.scene, interpolation='bicubic')
background.transform = scene.transforms.STTransform(scale=(500, 500), translate=(0, 0, -2000))

sun = scene.visuals.create_visual_node(SphereVisual)(radius=600, method='latitude', parent=view.scene, color=(1.0, 0.5, 0.2, 1.0))
sun.transform = scene.transforms.STTransform(translate=(0, 300, 2000))

# Grid Floor
grid = scene.visuals.GridLines(color=(1.0, 0.2, 1.0, 1.0), parent=view.scene)
grid.transform = scene.transforms.STTransform(scale=(10, 10, 1), translate=(0, -200, 1500))

# Function to generate mountain vertices and faces
def create_mountain_data(base_x, base_z, height, width, density=100):
    x = np.linspace(base_x - width / 2, base_x + width / 2, density)
    z = np.linspace(base_z - width / 2, base_z + width / 2, density)
    x, z = np.meshgrid(x, z)
    y = height * np.exp(-((x - base_x) ** 2 + (z - base_z) ** 2) / (2 * (width / 3) ** 2))
    vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    faces = []
    for i in range(density - 1):
        for j in range(density - 1):
            idx = i * density + j
            faces.append([idx, idx + 1, idx + density])
            faces.append([idx + 1, idx + density + 1, idx + density])
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces

# Create mountains
asks_mountain_vertices, asks_mountain_faces = create_mountain_data(-500, 1500, 600, 800)
bids_mountain_vertices, bids_mountain_faces = create_mountain_data(500, 1500, 600, 800)

asks_mountain = scene.visuals.Mesh(vertices=asks_mountain_vertices, faces=asks_mountain_faces, color=(0.2, 0.2, 1.0, 1.0), parent=view.scene)
bids_mountain = scene.visuals.Mesh(vertices=bids_mountain_vertices, faces=bids_mountain_faces, color=(0.2, 0.2, 1.0, 1.0), parent=view.scene)

# Adjust mountain transforms using MatrixTransform
asks_mountain_transform = MatrixTransform()
asks_mountain_transform.translate((-300, 600, 2000))  # Align asks mountain positively
asks_mountain_transform.scale((1.8, 2.5, 1))  # Scale height and width
asks_mountain_transform.rotate(90, (0, 1, 0))  # Rotate 90 degrees clockwise along the Y-axis
asks_mountain_transform.rotate(90, (1, 0, 0))  # Flip by 90 degrees along the X-axis
asks_mountain.transform = asks_mountain_transform

bids_mountain_transform = MatrixTransform()
bids_mountain_transform.translate((300, 600, 2000))  # Align bids mountain positively
bids_mountain_transform.scale((1.8, 2.5, 1))  # Scale height and width
bids_mountain_transform.rotate(-90, (0, 1, 0))  # Rotate -90 degrees counterclockwise along the Y-axis
bids_mountain_transform.rotate(90, (1, 0, 0))  # Flip by 90 degrees along the X-axis
bids_mountain.transform = bids_mountain_transform


# Timer for Updates
timer = app.Timer(interval=0.1, connect=update, start=True)

# Run WebSocket in Separate Thread
websocket_thread = threading.Thread(target=ws.run_forever)
websocket_thread.daemon = True
websocket_thread.start()

# Run Application
app.run()
