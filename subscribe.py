import numpy as np
from vispy import app, scene
from vispy.geometry import MeshData
from vispy.visuals import SphereVisual
from vispy.visuals.transforms import MatrixTransform
import websocket
import json
import time
import base64
import hashlib
import hmac
import os

# API Connection to environment variables
API_KEY = os.getenv('apikey')
API_SECRET = os.getenv('secretkey')

# Function to create a Kraken WebSocket signature
def get_signature(api_path, nonce, post_data, secret):
    message = (str(nonce) + post_data).encode()
    secret = base64.b64decode(secret)
    hashed = hmac.new(secret, message, hashlib.sha512)
    return base64.b64encode(hashed.digest()).decode()

# WebSocket message handler
current_price = 93600  # Replace with dynamic current price fetching logic
def on_message(ws, message):
    global current_price
    print("Message received:", message)
    try:
        data = json.loads(message)
        if isinstance(data, list) and len(data) > 1:
            if "a" in data[1]:
                asks = data[1]["a"]
                for ask in asks:
                    price, volume, timestamp = float(ask[0]), float(ask[1]), float(ask[2])
                    plot_mountain(price, volume, timestamp, quadrant=2)
            if "b" in data[1]:
                bids = data[1]["b"]
                for bid in bids:
                    price, volume, timestamp = float(bid[0]), float(bid[1]), float(bid[2])
                    plot_mountain(price, volume, timestamp, quadrant=1)
    except Exception as e:
        print("Error parsing message:", e)

# WebSocket connection open handler
def on_open(ws):
    # Generate nonce
    nonce = int(time.time() * 1000)

    # Define subscription message for public order book
    subscribe_data = {
        "event": "subscribe",
        "pair": ["XBT/USD"],
        "subscription": {"name": "book", "depth": 10}
    }

    # Send subscription message
    ws.send(json.dumps(subscribe_data))

# Connect to Kraken WebSocket API
url = "wss://ws.kraken.com/"  # Public WebSocket endpoint
ws = websocket.WebSocketApp(url, on_message=on_message, on_open=on_open)

# Canvas with 3D scene
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', show=True)
view = canvas.central_widget.add_view()

# POV of camera
view.camera = 'turntable'
view.camera.distance = 6700  # Move camera further back to view the scene
view.camera.center = (0, -500, 0)  # Center camera between Q3 and Q4 on Y-axis
view.camera.elevation = 15  # Flat view along the X-Z plane
view.camera.azimuth = 0  # Look straight ahead
view.camera.fov = 60  # Field of view

# Add a gradient background
background_data = np.linspace([0, 0, 0], [0.5, 0, 0.5], 500).astype('float32')
background = scene.visuals.Image(background_data.reshape(1, 500, 3),
                                 parent=canvas.scene, interpolation='bicubic')
background.transform = scene.transforms.STTransform(scale=(500, 500), translate=(0, 0, -2000))  # Push background further back

# Add a glowing 3D sun using a sphere mesh
sun = scene.visuals.create_visual_node(SphereVisual)(
    radius=900,  # Adjust the size of the sun
    method='latitude',  # Sphere construction method
    parent=view.scene,
    color=(1.0, 0.5, 0.2, 1.0),  # Glowing orange sun
)

# (x,z,y)
sun.transform = scene.transforms.STTransform(translate=(0, 5000, 100))  # Positive Z-axis far back

# Function to create axes for the quadrants
def create_axis_line(start, end, color, parent):
    pos = np.array([start, end])
    axis = scene.visuals.Line(pos=pos, color=color, parent=parent, width=2)
    return axis

# Create X, Y, Z axes for 3D quadrant layout
create_axis_line([-5000, 0, 0], [5000, 0, 0], (1, 0, 0, 1), view.scene)  # X-axis
create_axis_line([0, -5000, 0], [0, 5000, 0], (0, 1, 0, 1), view.scene)  # Y-axis
create_axis_line([0, 0, -5000], [0, 0, 5000], (0, 0, 1, 1), view.scene)  # Z-axis

# Add labels for axes
scene.visuals.Text(text='X', color='red', pos=(5500, 0, 0), parent=view.scene, font_size=40, bold=True)
scene.visuals.Text(text='Y', color='green', pos=(0, 5500, 0), parent=view.scene, font_size=40, bold=True)
scene.visuals.Text(text='Z', color='blue', pos=(0, 0, 5500), parent=view.scene, font_size=40, bold=True)

# Label quadrants
scene.visuals.Text(text='Q1', color='white', pos=(3000, 3000, 0), parent=view.scene, font_size=40, bold=True)
scene.visuals.Text(text='Q2', color='white', pos=(-3000, 3000, 0), parent=view.scene, font_size=40, bold=True)
scene.visuals.Text(text='Q3', color='white', pos=(-3000, -3000, 0), parent=view.scene, font_size=40, bold=True)
scene.visuals.Text(text='Q4', color='white', pos=(3000, -3000, 0), parent=view.scene, font_size=40, bold=True)


# Add a grid for the valley floor
grid = scene.visuals.GridLines(color=(1.0, 0.2, 1.0, 1.0), parent=view.scene)
grid.transform = scene.transforms.STTransform(scale=(10, 10, 1), translate=(0, -500, 0))  # Center grid at origin

# Function to plot mountains dynamically
def plot_mountain(price, volume, timestamp, quadrant):
    global current_price
    x_offset = price - current_price
    z_offset = -(timestamp % 5000)  # Wrap around to simulate progression

    if quadrant == 1:  # Quadrant 1 (Bids)
        x = abs(x_offset)
        y = volume
        z = z_offset
    elif quadrant == 2:  # Quadrant 2 (Asks)
        x = -abs(x_offset)
        y = volume
        z = z_offset

    vertex = np.array([[x, z, y]])
    mountain = scene.visuals.Markers()
    mountain.set_data(vertex, edge_color=None, face_color=(0.2, 0.8, 1.0, 1.0), size=5)
    mountain.parent = view.scene
# Function to generate mountain vertices and faces
# Function to generate mountain vertices and faces
def create_mountain_data(base_x, base_z, height, width, density=100):
    x = np.linspace(base_x - width / 2, base_x + width / 2, density)
    z = np.linspace(base_z - width / 2, base_z + width / 2, density)
    x, z = np.meshgrid(x, z)
    y = -height * np.exp(-((x - base_x) ** 2 + (z - base_z) ** 2) / (2 * (width / 3) ** 2))  # Make `y` negative to flip peaks upward
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
asks_mountain_vertices, asks_mountain_faces = create_mountain_data(-500, 1000, 600, 800)
bids_mountain_vertices, bids_mountain_faces = create_mountain_data(500, 1000, 600, 800)

def create_moving_mountain():
    vertices, faces = create_mountain_data(0, 2000, 400, 1200)
    mountain = scene.visuals.Mesh(vertices=vertices, faces=faces, color=(0.3, 0.8, 0.3, 1.0), parent=view.scene)
    mountain_transform = MatrixTransform()
    mountain_transform.translate((0, -500, 2000))
    mountain.transform = mountain_transform
    return mountain, mountain_transform

moving_mountain, moving_mountain_transform = create_moving_mountain()

asks_mountain = scene.visuals.Mesh(vertices=asks_mountain_vertices, faces=asks_mountain_faces, color=(0.2, 0.2, 1.0, 1.0), parent=view.scene)
bids_mountain = scene.visuals.Mesh(vertices=bids_mountain_vertices, faces=bids_mountain_faces, color=(0.2, 0.2, 1.0, 1.0), parent=view.scene)

# Adjust mountain transforms using MatrixTransform
asks_mountain_transform = MatrixTransform()
asks_mountain_transform.translate((-100, 500, 0))  # Negative X-axis, positive Z-axis
asks_mountain_transform.scale((1.8, 2.5, 1))  # Scale height and width
asks_mountain.transform = asks_mountain_transform

bids_mountain_transform = MatrixTransform()
bids_mountain_transform.translate((100, 500, 0))  # Positive X-axis, positive Z-axis
bids_mountain_transform.scale((1.8, 2.5, 1))  # Scale height and width
bids_mountain.transform = bids_mountain_transform

# Function to create wireframe data from mountain vertices and faces
def create_wireframe_data(vertices, faces):
    edges = set()
    for face in faces:
        edges.add(tuple(sorted((face[0], face[1]))))
        edges.add(tuple(sorted((face[1], face[2]))))
        edges.add(tuple(sorted((face[2], face[0]))))
    edges = np.array(list(edges))
    edge_vertices = vertices[edges]
    return edge_vertices

# Add wireframe for asks mountain
asks_wireframe_data = create_wireframe_data(asks_mountain_vertices, asks_mountain_faces)
asks_wireframe = scene.visuals.Line(
    pos=asks_wireframe_data.reshape(-1, 3),
    color=(0.8, 0.8, 0.8, 0.6),  # Light gray color with transparency
    parent=view.scene,
    connect='segments',
    width=0.5,
)
asks_wireframe.transform = asks_mountain_transform  # Apply the same transform as the mountain

# Add wireframe for bids mountain
bids_wireframe_data = create_wireframe_data(bids_mountain_vertices, bids_mountain_faces)
bids_wireframe = scene.visuals.Line(
    pos=bids_wireframe_data.reshape(-1, 3),
    color=(0.8, 0.8, 0.8, 0.6),  # Light gray color with transparency
    parent=view.scene,
    connect='segments',
    width=0.5,
)
bids_wireframe.transform = bids_mountain_transform  # Apply the same transform as the mountain

# Function to print mountain data
def print_mountain_data():
    global asks_mountain_vertices, bids_mountain_vertices
    print("Ask Mountain Vertices:")
    print(asks_mountain_vertices)
    print("\nBid Mountain Vertices:")
    print(bids_mountain_vertices)

# Real-time rendering update
def update(event):
    global bid_volumes, ask_volumes, moving_mountain_transform

    # Move the dynamic mountain range along the z-axis
    moving_mountain_transform.translate((0, 3000, -500))  # Start in center and move along Z-axis
    if moving_mountain_transform.matrix[2, 3] < -2000:  # Reset position when far back
        moving_mountain_transform.translate((0, 0, 4000))

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

    # Print mountain data periodically
    if int(time.time()) % 5 == 0:  # Print every 5 seconds
        print_mountain_data()

# Timer for updates
timer = app.Timer(interval=0.1, connect=update, start=True)

# Run WebSocket in a separate thread
import threading
websocket_thread = threading.Thread(target=ws.run_forever)
websocket_thread.daemon = True
websocket_thread.start()

# Run the application
app.run()
