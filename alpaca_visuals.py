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
        # Store raw bid and ask data snapshots
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
                messages = json.loads(raw)

                if not isinstance(messages, list):
                    print("Unexpected message format:", raw)
                    continue

                for message in messages:
                    if message.get("T") == "o":  # Order book update
                        bids = message.get("b", [])
                        asks = message.get("a", [])

                        # Update bids
                        for bid in bids:
                            price = float(bid["p"])
                            size = float(bid["s"])
                            if size == 0:
                                self.bid_data.pop(price, None)
                            else:
                                self.bid_data[price] = size

                        # Update asks
                        for ask in asks:
                            price = float(ask["p"])
                            size = float(ask["s"])
                            if size == 0:
                                self.ask_data.pop(price, None)
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

    def get_market_spread(self):
        """Calculate the spread between the lowest ask and the highest bid."""
        if self.bid_data and self.ask_data:
            highest_bid = max(self.bid_data.keys())
            lowest_ask = min(self.ask_data.keys())
            return lowest_ask - highest_bid
        return 0

    def record_current_snapshot(self, mid_price):
        """Store a snapshot of the current bid and ask data."""
        # Create deep copies to ensure historical data remains unchanged
        bid_copy = self.bid_data.copy()
        ask_copy = self.ask_data.copy()
        self.historical_depth.append((mid_price, bid_copy, ask_copy))


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
        self.camera_z_offset = 0
        self.camera_y_offset = 0
        self.wireframe_type = "rectangle"  # Initialize wireframe_type
        self._init_scene()
        self.timer = app.Timer(interval=0.1, connect=self.on_timer, start=True)

    def _init_scene(self):
        """Set up the scene visuals: sun, grid, wireframes, text labels."""
        # Sun Sphere
        self.sun = Sphere(radius=1000, method="latitude", parent=self.view.scene, color=(1.0, 0.5, 0.9, 1.0))
        self.sun.transform = scene.transforms.STTransform(translate=(0, 8000, 500))

        # Grid Lines
        self.grid = scene.visuals.GridLines(color=(1.0, 0.2, 1.0, 0.5), parent=self.view.scene)

        # Wireframe Lines
        self.batched_wireframe = scene.visuals.Line(parent=self.view.scene)

        # Current Price Label
        self.current_price_label = Text(
            text="?",
            color="white",
            font_size=18,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene
        )
        self.current_price_label.transform = scene.transforms.STTransform(translate=(10, 10))

        # Spread Label
        self.spread_label = Text(
            text="Spread: ?",
            color="white",
            font_size=18,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene
        )
        self.spread_label.transform = scene.transforms.STTransform(translate=(10, 40))

        # Connect key press event
        self.canvas.events.key_press.connect(self.on_key_press)

    def on_key_press(self, event):
        """Handle key press events to toggle wireframe type."""
        if event.key == "T":  # Switch to triangle or rectangle wireframe
            if self.wireframe_type == "rectangle":
                self.wireframe_type = "triangle"
            else:
                self.wireframe_type = "rectangle"
            print(f"Wireframe type switched to: {self.wireframe_type}")
            self.update_wireframe()  # Force update of the wireframe on toggle

    def on_timer(self, event):
        """Called periodically; updates the scene with data changes."""
        self.data.process_messages()
        mid_price = self.data.get_mid_price()
        if mid_price == 0:
            return

        self.current_price_label.text = f"{mid_price:.2f}"
        spread = self.data.get_market_spread()
        self.spread_label.text = f"Spread: {spread:.2f}"
        self.data.record_current_snapshot(mid_price)

        self.update_wireframe()

    def update_wireframe(self):
        """Rebuild the wireframe visualization based on the current wireframe_type."""
        verts = []
        cols = []
        volume_threshold = 0.5  # Threshold to filter out low-volume entries

        # Iterate through all historical snapshots
        for i, (mid_price, bid_data, ask_data) in enumerate(self.data.historical_depth):
            # Offset for layering historical data
            z_offset = (len(self.data.historical_depth) - 1 - i) * 5

            # Process Bids
            for price, volume in sorted(bid_data.items(), reverse=True)[:self.data.order_book_depth]:
                if volume < volume_threshold:
                    continue
                x = mid_price - price
                y = volume * 10
                color = [0.0, 1.0, 0.0, 1.0]  # Green for bids

                if self.wireframe_type == "rectangle":
                    verts.extend([
                        [x - 5, z_offset, 0],
                        [x + 5, z_offset, 0],
                        [x + 5, z_offset, y],
                        [x - 5, z_offset, y],
                        [x - 5, z_offset, 0],
                    ])
                    cols.extend([color] * 5)
                elif self.wireframe_type == "triangle":
                    verts.extend([
                        [x - 5, z_offset, 0],
                        [x + 5, z_offset, 0],
                        [x, z_offset, y],
                        [x - 5, z_offset, 0],  # Close the triangle
                    ])
                    cols.extend([color] * 4)

            # Process Asks
            for price, volume in sorted(ask_data.items())[:self.data.order_book_depth]:
                if volume < volume_threshold:
                    continue
                x = mid_price - price
                y = volume * 10
                color = [1.0, 0.0, 0.0, 1.0]  # Red for asks

                if self.wireframe_type == "rectangle":
                    verts.extend([
                        [x - 5, z_offset, 0],
                        [x + 5, z_offset, 0],
                        [x + 5, z_offset, y],
                        [x - 5, z_offset, y],
                        [x - 5, z_offset, 0],
                    ])
                    cols.extend([color] * 5)
                elif self.wireframe_type == "triangle":
                    verts.extend([
                        [x - 5, z_offset, 0],
                        [x + 5, z_offset, 0],
                        [x, z_offset, y],
                        [x - 5, z_offset, 0],  # Close the triangle
                    ])
                    cols.extend([color] * 4)

        if verts and cols:
            merged_verts = np.array(verts, dtype=np.float32)
            merged_cols = np.array(cols, dtype=np.float32)
            self.batched_wireframe.set_data(pos=merged_verts, color=merged_cols)

        self.canvas.update()

    def run(self):
        """Start the Vispy event loop."""
        app.run()


###############################################################################
# WebSocket Handlers
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
# Main Execution
###############################################################################
def main():
    load_dotenv()

    # Verify that Alpaca API credentials are set
    api_key = os.getenv('ALPACA_PAPER_API_KEY')
    api_secret = os.getenv('ALPACA_PAPER_API_SECRET')

    if not api_key or not api_secret:
        print("Error: Alpaca API credentials not found in environment variables.")
        return

    data_mgr = DataManager(max_snapshots=500, order_book_depth=1000)
    viz = VaporWaveOrderBookVisualizer(data_mgr)
    ws = build_websocket(data_mgr)
    threading.Thread(target=ws.run_forever, daemon=True).start()
    viz.run()


if __name__ == "__main__":
    main()
