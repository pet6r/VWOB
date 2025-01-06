import numpy as np
import json
import threading
from collections import deque
from queue import Queue

import websocket
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

        self.bid_data = {}
        self.ask_data = {}
        self.historical_depth = deque(maxlen=self.max_snapshots)

        self.previous_bid_data = {}
        self.previous_ask_data = {}
        self.previous_mid_price = None

        self.message_queue = Queue()

    def on_message(self, ws, raw_message):
        """Callback from WebSocket, push raw message to queue."""
        self.message_queue.put(raw_message)

    def process_messages(self):
        """Process all pending messages to update bid_data/ask_data."""
        while not self.message_queue.empty():
            raw = self.message_queue.get()
            try:
                data = json.loads(raw)
                if "b" in data and "a" in data:
                    # Parse bids
                    for bid in data["b"]:
                        price, volume = float(bid[0]), float(bid[1])
                        if volume == 0:
                            self.bid_data.pop(price, None)
                        else:
                            self.bid_data[price] = volume
                    # Parse asks
                    for ask in data["a"]:
                        price, volume = float(ask[0]), float(ask[1])
                        if volume == 0:
                            self.ask_data.pop(price, None)
                        else:
                            self.ask_data[price] = volume
            except Exception as e:
                print("Error processing message:", e)

    def get_mid_price(self):
        """Compute mid price from highest bid and lowest ask."""
        if self.bid_data and self.ask_data:
            highest_bid = max(self.bid_data.keys())
            lowest_ask = min(self.ask_data.keys())
            return (highest_bid + lowest_ask) / 2
        return 0

    def has_changed(self):
        """Return True if the order book changed since the last update."""
        return (self.bid_data != self.previous_bid_data or
                self.ask_data != self.previous_ask_data)

    def record_current_snapshot(self, mid_price):
        """Build geometry for the current snapshot, store in historical_depth."""
        verts, cols = self.build_wireframe_geometry(mid_price)
        self.historical_depth.append((verts, cols))  # Retain correct colors for this snapshot

        self.previous_bid_data = dict(self.bid_data)
        self.previous_ask_data = dict(self.ask_data)
        self.previous_mid_price = mid_price


    def build_wireframe_geometry(self, mid_price):
        """Create vertices and colors for the current snapshot."""
        verts = []
        cols = []

        # Bids
        for price, volume in sorted(self.bid_data.items(), reverse=True)[:self.order_book_depth]:
            x = mid_price - price
            y = volume * 10
            color = [0.0, 1.0, 0.0, 1.0] if volume <= self.volume_spike_threshold else [0.2, 1.0, 0.2, 1.0]  # Green
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
            color = [1.0, 0.0, 0.0, 1.0] if volume <= self.volume_spike_threshold else [1.0, 0.2, 0.2, 1.0]  # Red
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
        total_bid_volume = sum(self.bid_data.values())
        total_ask_volume = sum(self.ask_data.values())

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
        # Big "Sun"
        self.sun = Sphere(radius=2000, method="latitude", parent=self.view.scene, color=(1.0, 0.5, 0.9, 1.0))
        self.sun.transform = scene.transforms.STTransform(translate=(0, 8000, 800))

        # Grid
        self.grid = scene.visuals.GridLines(color=(1.0, 0.2, 1.0, 0.5), parent=self.view.scene)
        self.grid.transform = scene.transforms.STTransform(scale=(5, 1, 5), translate=(0, 0, 0))

        # Vapor Plane
        plane_size = 20000
        plane_data = np.zeros((512, 512, 4), dtype=np.float32)
        for i in range(512):
            for j in range(512):
                t = float(i) / 511
                plane_data[i, j, 0] = t
                plane_data[i, j, 1] = 0
                plane_data[i, j, 2] = t
                plane_data[i, j, 3] = 1.0
        self.plane_tex = scene.visuals.Image(data=plane_data, parent=self.view.scene, interpolation='linear') # xzy
        self.plane_tex.transform = scene.transforms.STTransform(translate=(-plane_size/2, -plane_size/2, 0), scale=(plane_size/512, plane_size/512, 1)
        )

        # Main wireframe
        self.batched_wireframe = scene.visuals.Line(parent=self.view.scene)

        # Tron bloom: ghost wireframe
        self.ghost_wireframe = scene.visuals.Line(parent=self.view.scene)
        self.ghost_wireframe.set_gl_state(blend=True, depth_test=True, blend_func=("src_alpha", "one"))

        # WHITE OUTLINE: Latest snapshot highlight
        self.outline_wireframe = scene.visuals.Line(parent=self.view.scene)
        self.outline_wireframe.set_gl_state(
            blend=True,
            depth_test=True,
            line_width=5.0  # Slightly thicker line for better visibility
        )

        # Current price label
        self.current_price_label = Text(
            text="?", color="white", font_size=18, anchor_x="left", anchor_y="bottom", parent=self.canvas.scene)
        self.current_price_label.transform = scene.transforms.STTransform(translate=(10, 10))

        # Market control label
        self.control_label = Text(
            text="Neutral", color="white", font_size=10, anchor_x="left", anchor_y="bottom", parent=self.canvas.scene
        )
        self.control_label.transform = scene.transforms.STTransform(translate=(10, 50))  # Position near top-left

    def on_timer(self, event):
        """Called periodically; updates the scene with data changes."""
        # 1) Let DataManager process messages
        self.data.process_messages()

        # 2) Check if anything changed in the order book
        if not self.data.has_changed():
            return  # No update needed, skip

        # 3) Rebuild geometry if changed
        mid_price = self.data.get_mid_price()
        self.data.record_current_snapshot(mid_price)

        # 4) Combine geometry from historical_depth with Z-offset
        all_verts = []
        all_cols = []
        for i, (snap_verts, snap_cols) in enumerate(self.data.historical_depth):
            shifted = snap_verts.copy()
            shifted[:, 1] += (len(self.data.historical_depth) - 1 - i) * 5  # Offset in Z (third) axis
            all_verts.append(shifted)
            all_cols.append(snap_cols)

        merged_verts = np.concatenate(all_verts, axis=0) if all_verts else np.zeros((0, 3))
        merged_cols = np.concatenate(all_cols, axis=0) if all_cols else np.zeros((0, 4))

        # 5) Update main wireframe
        self.batched_wireframe.set_data(pos=merged_verts, color=merged_cols)

        # Tron bloom: ghost wireframe
        ghost_verts = merged_verts * 1.01  # Slightly bigger
        ghost_cols = merged_cols.copy()
        ghost_cols[:, 3] *= 0.3  # Reduce alpha for ghosting
        self.ghost_wireframe.set_data(pos=ghost_verts, color=ghost_cols)

        # 6) Update price label
        self.current_price_label.text = f"{float(mid_price):.2f}"

        # 7) White Outline for Latest Snapshot
        if self.data.historical_depth:
            newest_verts, newest_cols = self.data.historical_depth[-1]
            outline_verts = newest_verts.copy()
            outline_cols = np.ones_like(newest_cols)
            outline_cols[:, :3] = 1.0  # Pure white
            outline_cols[:, 3] = 1.0  # Full opacity
            self.outline_wireframe.set_data(pos=outline_verts, color=outline_cols)

        # 8) Update market control label
        market_control = self.data.get_market_control()
        self.control_label.text = market_control

        # 9) Spin camera slightly for dynamic effect
        self.view.camera.azimuth += 0.05
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
        "params": [f"{symbol}@depth@1000ms"],
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
    # Create data manager
    data_mgr = DataManager(
        max_snapshots=500,
        order_book_depth=1000,
        volume_spike_threshold=61
    )

    # Create visualizer referencing that data
    viz = VaporWaveOrderBookVisualizer(data_mgr)

    ws = build_websocket(data_mgr)
    threading.Thread(target=ws.run_forever, daemon=True).start()
    viz.run()

if __name__ == "__main__":
    main()
