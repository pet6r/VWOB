import numpy as np
import json
import threading
from collections import deque
from queue import Queue
from vispy.util.transforms import translate
import websocket
from dotenv import load_dotenv
import os
from vispy import app, scene
from vispy.scene.visuals import Text, Sphere, Image

###############################################################################
# DataManager: Manages incoming order book data, stores bids/asks, provides geometry
###############################################################################
class DataManager:
    def __init__(self, max_snapshots=500, order_book_depth=1000):
        self.max_snapshots = max_snapshots
        self.order_book_depth = order_book_depth
        self.bid_data = {}
        self.ask_data = {}
        self.historical_depth = deque(maxlen=self.max_snapshots)
        self.message_queue = Queue()

    def on_message(self, ws, raw_message):
        """Callback from WebSocket, push raw message to queue."""
        self.message_queue.put(raw_message)

    def process_messages(self):
        """Process all pending messages to update bid_data/ask_data."""
        while not self.message_queue.empty():
            raw = self.message_queue.get()
            try:
                #print("Raw message:", raw)  # Log raw messages for debugging
                messages = json.loads(raw)

                if not isinstance(messages, list):  # Expecting a list of messages
                    print("Unexpected message format:", raw)
                    continue

                for message in messages:
                    if message.get("T") == "o":  # Order book update
                        bids = message.get("b", [])  # List of bids
                        asks = message.get("a", [])  # List of asks

                        # Update bids
                        for bid in bids:
                            price = float(bid["p"])  # Correct parsing for price
                            size = float(bid["s"])  # Correct parsing for size
                            if size == 0:
                                self.bid_data.pop(price, None)  # Remove price level
                            else:
                                self.bid_data[price] = size

                        # Update asks
                        for ask in asks:
                            price = float(ask["p"])  # Correct parsing for price
                            size = float(ask["s"])  # Correct parsing for size
                            if size == 0:
                                self.ask_data.pop(price, None)  # Remove price level
                            else:
                                self.ask_data[price] = size

            except json.JSONDecodeError:
                print("Error decoding JSON message:", raw)
            except KeyError as e:
                print(f"Missing expected key in message: {e}. Message: {raw}")
            except Exception as e:
                print("Error processing message:", e)


    def get_mid_price(self):
        """Compute mid price from highest bid and lowest ask."""
        if self.bid_data and self.ask_data:
            highest_bid = max(self.bid_data.keys())
            lowest_ask = min(self.ask_data.keys())
            return (highest_bid + lowest_ask) / 2
        return 0


    def record_current_snapshot(self, mid_price):
        """Build geometry for the current snapshot, store in historical_depth."""
        verts, cols = self.build_wireframe_geometry(mid_price)
        self.historical_depth.append((verts, cols))

    def build_wireframe_geometry(self, mid_price):
        """Create vertices and colors for the current snapshot."""
        verts = []
        cols = []

        # Bids
        for price, volume in sorted(self.bid_data.items(), reverse=True)[:self.order_book_depth]:
            x = mid_price - price
            y = volume * 10
            color = [0.0, 1.0, 0.0, 1.0]  # Green
            verts.extend([
                [x - 5, 0, 0],
                [x + 5, 0, 0],
                [x + 5, 0, y],
                [x - 5, 0, y],
                [x - 5, 0, 0],
            ])
            cols.extend([color] * 5)

        # Asks
        for price, volume in sorted(self.ask_data.items())[:self.order_book_depth]:
            x = mid_price - price
            y = volume * 10
            color = [1.0, 0.0, 0.0, 1.0]  # Red
            verts.extend([
                [x - 5, 0, 0],
                [x + 5, 0, 0],
                [x + 5, 0, y],
                [x - 5, 0, y],
                [x - 5, 0, 0],
            ])
            cols.extend([color] * 5)

        return np.array(verts, dtype=np.float32), np.array(cols, dtype=np.float32)

###############################################################################
# VaporWaveOrderBookVisualizer: Builds the scene and updates from DataManager
###############################################################################
class VaporWaveOrderBookVisualizer:
    def __init__(self, data_manager):
        self.data = data_manager
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="#220033", show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.distance = 1500
        self.view.camera.center = (0, 0, 0)
        self.view.camera.elevation = 20
        self.view.camera.fov = 60
        self._init_scene()
        self.timer = app.Timer(interval=0.1, connect=self.on_timer, start=True)
        self.camera_z_offset = 0  # Initialize Z-axis offset
        self.camera_y_offset = 0  # Initialize Y-axis offset

    def _init_scene(self):
        """Set up the scene visuals: sun, grid, plane, wireframes, text label."""
        self.sun = Sphere(radius=2000, method="latitude", parent=self.view.scene, color=(1.0, 0.5, 0.9, 1.0))
        self.sun.transform = scene.transforms.STTransform(translate=(0, 8000, 800))
        self.grid = scene.visuals.GridLines(color=(1.0, 0.2, 1.0, 0.5), parent=self.view.scene)
        self.batched_wireframe = scene.visuals.Line(parent=self.view.scene)

        self.current_price_label = Text(
            text="?",  # Placeholder
            color="white",
            font_size=18,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene  # Overlay on 2D canvas
        )
        self.current_price_label.transform = scene.transforms.STTransform(translate=(10, 10))

        # Connect key press events
        self.canvas.events.key_press.connect(self.on_key_press)


    def on_key_press(self, event):
            """Move camera center with arrow keys and update display."""
            step = 10  # The distance to move the camera with each key press
            if event.key == 'Left':
                self.camera_z_offset -= step
            elif event.key == 'Right':
                self.camera_z_offset += step
            elif event.key == 'Up':
                self.camera_y_offset += step
            elif event.key == 'Down':
                self.camera_y_offset -= step

            # Update camera center
            old_center = list(self.view.camera.center)
            old_center[1] = self.camera_z_offset  # Update Z-axis offset
            old_center[2] = self.camera_y_offset  # Update Y-axis offset
            self.view.camera.center = tuple(old_center)

            # Update the label to show the current camera offsets
            #self.current_price_label.text = (            )
            self.canvas.update()

    def on_timer(self, event):
        """Called periodically; updates the scene with data changes."""
        # Process incoming messages
        self.data.process_messages()

        # Get the mid price and record a new snapshot if there's valid data
        mid_price = self.data.get_mid_price()
        if mid_price == 0:  # Skip if there's no valid mid price
            return

        self.current_price_label.text = f"{mid_price:.2f}"  # Updated price with 2 decimal places


        self.data.record_current_snapshot(mid_price)

        # Combine geometry from historical_depth with Z-offset
        all_verts = []
        all_cols = []
        for i, (snap_verts, snap_cols) in enumerate(self.data.historical_depth):
            if snap_verts.size == 0 or snap_cols.size == 0:  # Skip empty snapshots
                continue
            shifted = snap_verts.copy()
            shifted[:, 1] += (len(self.data.historical_depth) - 1 - i) * 5  # Offset in Z (third) axis
            all_verts.append(shifted)
            all_cols.append(snap_cols)

        # Ensure there is data to visualize
        if all_verts and all_cols:
            merged_verts = np.concatenate(all_verts, axis=0)
            merged_cols = np.concatenate(all_cols, axis=0)
            self.batched_wireframe.set_data(pos=merged_verts, color=merged_cols)

        # Update the canvas
        self.canvas.update()

    def run(self):
        app.run()

###############################################################################
# WebSocket
###############################################################################
def on_open(ws):
    print("Authenticating with Alpaca...")
    auth_payload = {
        "action": "auth",
        "key": os.getenv('ALPACA_PAPER_API_KEY'),
        "secret": os.getenv('ALPACA_PAPER_API_SECRET')
    }
    ws.send(json.dumps(auth_payload))

    print("Subscribing to order book updates...")
    subscription_payload = {
        "action": "subscribe",
        "orderbooks": ["BTC/USD"]
    }
    ws.send(json.dumps(subscription_payload))


def on_error(ws, error):
    print("WebSocket error:", error)


def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed:", close_status_code, close_msg)


def build_websocket(data_mgr):
    return websocket.WebSocketApp(
        "wss://stream.data.alpaca.markets/v1beta3/crypto/us",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
        on_message=data_mgr.on_message
    )

###############################################################################
# Main
###############################################################################
def main():
    load_dotenv()
    data_mgr = DataManager(max_snapshots=500, order_book_depth=1000)
    viz = VaporWaveOrderBookVisualizer(data_mgr)
    ws = build_websocket(data_mgr)
    threading.Thread(target=ws.run_forever, daemon=True).start()
    viz.run()

if __name__ == "__main__":
    main()
