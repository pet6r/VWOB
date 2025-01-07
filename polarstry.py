import polars as pl
import numpy as np
import json
import websocket
import threading
from queue import Queue
from vispy import app, scene
from vispy.scene.visuals import Text, Sphere, Image

###############################################################################
# DataManager: Manages incoming book data, stores bids/asks, provides geometry
###############################################################################
class DataManager:
    def __init__(self, max_snapshots=500, order_book_depth=1000, volume_spike_threshold=61):
        self.max_snapshots = max_snapshots
        self.order_book_depth = order_book_depth
        self.volume_spike_threshold = volume_spike_threshold

        self.bid_data = pl.DataFrame({"price": [], "volume": []}, schema={"price": pl.Float64, "volume": pl.Float64})
        self.ask_data = pl.DataFrame({"price": [], "volume": []}, schema={"price": pl.Float64, "volume": pl.Float64})
        self.historical_depth = []  # Store geometry snapshots as a list

        self.previous_mid_price = None
        self.message_queue = Queue()

    def on_message(self, ws, raw_message):
        """Callback from WebSocket, push raw message to queue."""
        self.message_queue.put(raw_message)

    def sanitize_data(self, data):
        """Ensure incoming data adheres to expected schema."""
        sanitized = []
        for item in data:
            try:
                if len(item) == 2:
                    price = float(item[0])
                    volume = float(item[1])
                    sanitized.append({"price": price, "volume": volume})
                else:
                    raise ValueError("Item does not have exactly two elements.")
            except (ValueError, IndexError, TypeError) as e:
                print(f"Skipping malformed data item {item}: {e}")
        return sanitized

    def process_messages(self):
        """Process all pending messages to update bid_data/ask_data."""
        while not self.message_queue.empty():
            raw = self.message_queue.get()
            try:
                data = json.loads(raw)
                if "b" in data and "a" in data:
                    # Sanitize and update bids
                    bid_updates = pl.DataFrame(
                        self.sanitize_data(data.get("b", [])),
                        schema={"price": pl.Float64, "volume": pl.Float64}
                    )
                    self.bid_data = (
                        self.bid_data.vstack(bid_updates)
                        .filter(pl.col("volume") > 0)  # Remove zero-volume rows
                        .unique(subset="price", keep="last")  # Keep latest updates per price
                    )

                    # Sanitize and update asks
                    ask_updates = pl.DataFrame(
                        self.sanitize_data(data.get("a", [])),
                        schema={"price": pl.Float64, "volume": pl.Float64}
                    )
                    self.ask_data = (
                        self.ask_data.vstack(ask_updates)
                        .filter(pl.col("volume") > 0)  # Remove zero-volume rows
                        .unique(subset="price", keep="last")  # Keep latest updates per price
                    )

                    # Debug: Print updated bid and ask data
                    print("Updated Bids:", self.bid_data)
                    print("Updated Asks:", self.ask_data)
            except Exception as e:
                print(f"Error processing message: {e}\nRaw message: {raw}")

    def get_mid_price(self):
        """Compute mid price from highest bid and lowest ask."""
        try:
            if len(self.bid_data) > 0 and len(self.ask_data) > 0:
                highest_bid = self.bid_data["price"].max()
                lowest_ask = self.ask_data["price"].min()
                print(f"Highest Bid: {highest_bid}, Lowest Ask: {lowest_ask}")
                return (highest_bid + lowest_ask) / 2
        except Exception as e:
            print(f"Error calculating mid price: {e}")
        return 0

    def record_current_snapshot(self, mid_price):
        """Build geometry for the current snapshot, store in historical_depth."""
        verts, cols = self.build_wireframe_geometry(mid_price)
        self.historical_depth.append({"mid_price": mid_price, "verts": verts, "cols": cols})
        if len(self.historical_depth) > self.max_snapshots:
            self.historical_depth.pop(0)

    def build_wireframe_geometry(self, mid_price):
        """Create vertices and colors for the current snapshot."""
        verts = []
        cols = []

        # Bids
        for price, volume in zip(self.bid_data["price"].to_list(), self.bid_data["volume"].to_list()):
            x = mid_price - price
            y = volume * 10
            color = [0.0, 1.0, 0.0, 1.0] if volume <= self.volume_spike_threshold else [0.2, 1.0, 0.2, 1.0]
            verts.extend([
                [x - 5, 0, 0],
                [x + 5, 0, 0],
                [x + 5, 0, y],
                [x - 5, 0, y],
                [x - 5, 0, 0],
            ])
            cols.extend([color] * 5)

        # Asks
        for price, volume in zip(self.ask_data["price"].to_list(), self.ask_data["volume"].to_list()):
            x = mid_price - price
            y = volume * 10
            color = [1.0, 0.0, 0.0, 1.0] if volume <= self.volume_spike_threshold else [1.0, 0.2, 0.2, 1.0]
            verts.extend([
                [x - 5, 0, 0],
                [x + 5, 0, 0],
                [x + 5, 0, y],
                [x - 5, 0, y],
                [x - 5, 0, 0],
            ])
            cols.extend([color] * 5)

        return np.array(verts, dtype=np.float32), np.array(cols, dtype=np.float32)

    def get_market_control(self):
        """Determine whether buyers or sellers are in control."""
        total_bid_volume = self.bid_data["volume"].sum()
        total_ask_volume = self.ask_data["volume"].sum()

        if total_bid_volume > total_ask_volume:
            return "Buyers in Control"
        elif total_ask_volume > total_bid_volume:
            return "Sellers in Control"
        else:
            return "Neutral"

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

        self.timer = app.Timer(interval=0.25, connect=self.on_timer, start=True)

    def _init_scene(self):
        """Set up the scene visuals: sun, grid, plane, wireframes, text label."""
        self.sun = Sphere(radius=2000, method="latitude", parent=self.view.scene, color=(1.0, 0.5, 0.9, 1.0))
        self.grid = scene.visuals.GridLines(color=(1.0, 0.2, 1.0, 0.5), parent=self.view.scene)
        self.batched_wireframe = scene.visuals.Line(parent=self.view.scene)
        self.current_price_label = Text(
            text="?", color="white", font_size=18, anchor_x="left", anchor_y="bottom", parent=self.canvas.scene
        )

    def on_timer(self, event):
        """Called periodically; updates the scene with data changes."""
        self.data.process_messages()

        if len(self.data.bid_data) == 0 or len(self.data.ask_data) == 0:
            print("Waiting for sufficient data to calculate mid price...")
            return

        mid_price = self.data.get_mid_price()
        self.data.record_current_snapshot(mid_price)

        all_verts = []
        all_cols = []
        for snapshot in self.data.historical_depth:
            verts = snapshot["verts"]
            cols = snapshot["cols"]
            all_verts.append(verts)
            all_cols.append(cols)

        self.batched_wireframe.set_data(pos=np.concatenate(all_verts), color=np.concatenate(all_cols))
        self.current_price_label.text = f"{mid_price:.2f}"
        self.canvas.update()

    def run(self):
        app.run()

###############################################################################
# WebSocket
###############################################################################
def on_open(ws):
    symbol = "btcusdt"
    params = {
        "method": "SUBSCRIBE",
        "params": [f"{symbol}@depth@100ms"],
        "id": 1
    }
    ws.send(json.dumps(params))
    print("Subscribed to Binance order book")

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, status_code, msg):
    print("WebSocket closed:", status_code, msg)

def build_websocket(data_mgr):
    return websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
        on_message=data_mgr.on_message
    )

###############################################################################
# Main
###############################################################################
def main():
    data_mgr = DataManager(
        max_snapshots=500,
        order_book_depth=1000,
        volume_spike_threshold=61
    )

    viz = VaporWaveOrderBookVisualizer(data_mgr)

    ws = build_websocket(data_mgr)
    threading.Thread(target=ws.run_forever, daemon=True).start()
    viz.run()

if __name__ == "__main__":
    main()
