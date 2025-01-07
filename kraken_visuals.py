import numpy as np
import json
import threading
from collections import deque
from queue import Queue

import websocket
from vispy import app, scene
from vispy.scene.visuals import Text, Sphere, Image

###############################################################################
# DataManager: Manages incoming book data, store bids/asks, provide geometry
###############################################################################
class DataManager:

    def __init__(self, max_snapshots=1000, order_book_depth=1000, volume_spike_threshold=61):
        self.max_snapshots = max_snapshots
        self.order_book_depth = order_book_depth
        self.volume_spike_threshold = volume_spike_threshold #look for extremes

        self.bid_data = {} # webconn data for bids/buys
        self.ask_data = {} # webconn data for asks/sells
        self.historical_depth = deque(maxlen=self.max_snapshots) # memory of depth data

        self.previous_bid_data = {}
        self.previous_ask_data = {}
        self.previous_mid_price = None # current price

        self.message_queue = Queue()

    def on_message(self, ws, msg):
        """Callback from websocket, push raw message to queue."""
        self.message_queue.put(msg)

    def process_messages(self):
        """Process all pending messages to update self.bid_data / self.ask_data."""
        while not self.message_queue.empty():
            raw = self.message_queue.get()
            try:
                data = json.loads(raw)
                # We expect data like: [channel_id, { "b": [...], "a": [...] }, ...]
                if isinstance(data, list) and len(data) > 1:
                    # parse bids
                    for b in data[1].get("b", []):
                        price, volume = float(b[0]), float(b[1])
                        if volume == 0:
                            self.bid_data.pop(price, None)
                        else:
                            self.bid_data[price] = volume

                    # parse asks
                    for a in data[1].get("a", []):
                        price, volume = float(a[0]), float(a[1])
                        if volume == 0:
                            self.ask_data.pop(price, None)
                        else:
                            self.ask_data[price] = volume
            except Exception as e:
                print("Error processing message:", e)

    def get_mid_price(self):
        """Compute mid price from highest bid, lowest ask."""
        if self.bid_data and self.ask_data:
            highest_bid = max(self.bid_data.keys())
            lowest_ask  = min(self.ask_data.keys())
            return (highest_bid + lowest_ask) / 2
        return 0

    def has_changed(self):
        """Return True if the order book changed since last update."""
        changed = (self.bid_data != self.previous_bid_data or
                   self.ask_data != self.previous_ask_data)
        return changed

    def record_current_snapshot(self, mid_price):
        """Build geometry for the current snapshot, store in historical_depth."""
        verts, cols = self.build_wireframe_geometry(mid_price)
        self.historical_depth.append((verts, cols))

        # Save current state for next check
        self.previous_bid_data = dict(self.bid_data)
        self.previous_ask_data = dict(self.ask_data)
        self.previous_mid_price = mid_price

    def build_wireframe_geometry(self, mid_price):
        """Create vertices + colors for the current snapshot from self.bid_data / self.ask_data."""
        verts = []
        cols  = []

        # Bids
        for price, volume in sorted(self.bid_data.items(), reverse=True)[:self.order_book_depth]:
            x = mid_price - price
            y = volume * 10
            if volume > self.volume_spike_threshold:
                color = [0.8, 1.0, 0.2, 0.2]  # Neon green/yellow with 20% transparency
            else:
                color = [0, 1, 0.6, 1]
            verts.extend([
                [x - 5, 0,   0],
                [x + 5, 0,   0],
                [x + 5, 0,   y],
                [x - 5, 0,   y],
                [x - 5, 0,   0],
            ])
            cols.extend([color]*5)

        # Asks
        for price, volume in sorted(self.ask_data.items())[:self.order_book_depth]:
            x = mid_price - price
            y = volume * 10
            if volume > self.volume_spike_threshold:
                color = [1.0, 0.2, 0.6, 0.2] # hot red/pink with opacity
            else:
                color = [1, 0.2, 0.8, 1]
            verts.extend([
                [x - 5, 0,   0],
                [x + 5, 0,   0],
                [x + 5, 0,   y],
                [x - 5, 0,   y],
                [x - 5, 0,   0],
            ])
            cols.extend([color]*5)

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
# VaporWaveOrderBookVisualizer: Builds the scene & updates it from a DataManager
###############################################################################
class VaporWaveOrderBookVisualizer:
    def __init__(self, data_manager):
        self.data = data_manager

        # create the scene canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            bgcolor='#220033',
            show=True,
            config={'depth_size': 24, 'samples': 8}  # Enable 8x MSAA (multi-sample anti-aliasing)

        )
        self.view = self.canvas.central_widget.add_view()

        # camera
        self.view.camera = 'turntable'
        self.view.camera.distance = 1500
        self.view.camera.center   = (0, 0, 0)  # (x, z, y)
        self.view.camera.elevation = 20
        self.view.camera.azimuth   = 0
        self.view.camera.fov       = 60

        # create visuals
        self._init_scene()

        # timer controlling our main update loop
        self.timer = app.Timer(interval=0.1, connect=self.on_timer, start=True)

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

        # main wireframe
        self.batched_wireframe = scene.visuals.Line(parent=self.view.scene)

        # bloom "ghost" wireframe
        self.ghost_wireframe = scene.visuals.Line(parent=self.view.scene)
        self.ghost_wireframe.set_gl_state(
            blend=True,
            depth_test=True,
            blend_func=('src_alpha', 'one')  # additive blending
        )

        # WHITE OUTLINE: brand-new wireframe dedicated to outlining the newest snapshot
        self.outline_wireframe = scene.visuals.Line(parent=self.view.scene)
        self.outline_wireframe.set_gl_state(
            blend=True,
            depth_test=True,
            line_width=5.0  # slightly thicker line
        )

        # label for "The Mirror Stage"
        self.mirror_label = Text(text="^", color='white', font_size=2000, parent=self.view.scene, anchor_x='center', anchor_y='center', bold=True)
        self.mirror_label.transform = scene.transforms.STTransform(translate=(0, 0, -2)) # xzy

        # current price label at top-left of screen
        self.current_price_label = Text(
            text="?",
            color="white",
            font_size=18,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene  # 2D overlay
        )
        self.current_price_label.transform = scene.transforms.STTransform(translate=(10, 10))

        # arrow-key movement offsets
        self.camera_z_offset = 0
        self.camera_y_offset = 0
        self.canvas.events.key_press.connect(self.on_key_press)

        # for the market control label
        self.control_label = Text(
            text="Neutral",
            color="white",
            font_size=10,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene
        )
        self.control_label.transform = scene.transforms.STTransform(translate=(10, 50))  # Position near the top-left


    def on_key_press(self, event):
        """Move camera center with arrow keys."""
        step = 10
        if event.key == 'Left':
            self.camera_z_offset -= step
        elif event.key == 'Right':
            self.camera_z_offset += step
        elif event.key == 'Up':
            self.camera_y_offset += step
        elif event.key == 'Down':
            self.camera_y_offset -= step

        old_center = list(self.view.camera.center)
        old_center[1] = self.camera_z_offset
        old_center[2] = self.camera_y_offset
        self.view.camera.center = tuple(old_center)
        self.canvas.update()

    def on_timer(self, event):
        """Called periodically; let's see if data changed, then update scene."""
        # 1) Let DataManager process messages
        self.data.process_messages()

        # 2) Check if anything changed in the order book
        if not self.data.has_changed():
            return  # no update needed, skip

        # 3) Rebuild geometry if changed
        mid_price = self.data.get_mid_price()
        self.data.record_current_snapshot(mid_price)

        # 4) Combine geometry from historical_depth
        all_verts = []
        all_cols  = []
        for i, (snap_verts, snap_cols) in enumerate(self.data.historical_depth):
            shifted = snap_verts.copy()
            # shift in +z => second coord
            shifted[:,1] += (len(self.data.historical_depth)-1 - i)*5
            all_verts.append(shifted)
            all_cols.append(snap_cols)

        merged_verts = np.concatenate(all_verts, axis=0) if all_verts else np.zeros((0,3))
        merged_cols  = np.concatenate(all_cols, axis=0) if all_cols else np.zeros((0,4))

        # 5) Update main wireframe
        self.batched_wireframe.set_data(pos=merged_verts, color=merged_cols)

        # Tron bloom: ghost wireframe
        ghost_verts = merged_verts.copy() * 1.01  # slightly bigger
        ghost_cols  = merged_cols.copy()
        ghost_cols[:,3] *= 0.3  # reduce alpha
        self.ghost_wireframe.set_data(pos=ghost_verts, color=ghost_cols)

        # 6) Update price label
        self.current_price_label.text = f"{float(mid_price)}"

        # 7) White Outline For Latest Snapshot
        if self.data.historical_depth:
            newest_verts, newest_cols = self.data.historical_depth[-2]
            outline_verts = newest_verts.copy()
            outline_cols  = np.ones_like(newest_cols)
            # ARGB => set R,G,B=1, alpha=1 => pure white
            outline_cols[:, :3] = 1.0
            outline_cols[:, 3]  = 1.0

            # shift it just like we do for the newest snapshot => i=last => shift=0
            # or if you'd like an offset, you can do something else
            # for consistency, let's shift it by 0
            # (no shift needed since the newest is shifted by 0 above)
            self.outline_wireframe.set_data(pos=outline_verts, color=outline_cols)

        # 8) Update control label
            market_control = self.data.get_market_control()
            self.control_label.text = market_control

        # 9) Spin camera a bit
        self.view.camera.azimuth += 0.05
        self.canvas.update()

    def run(self):
        """Start the Vispy event loop."""
        app.run()


###############################################################################
# WebSocket
###############################################################################
def on_open(ws):
    subscription = {
        "event": "subscribe",
        "pair": ["BTC/USDT"],
        "subscription": {"name": "book", "depth": 1000}
    }
    ws.send(json.dumps(subscription))
    print("Subscribed to order book")

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, status_code, msg):
    print("WebSocket closed:", status_code, msg)

def build_websocket(data_mgr):
    """Create the WebSocketApp, hooking data_mgr's callbacks."""
    return websocket.WebSocketApp(
        "wss://ws.kraken.com/",
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
        max_snapshots=1500,
        order_book_depth=1000,
        volume_spike_threshold=61
    )

    # Create visualizer referencing that data
    viz = VaporWaveOrderBookVisualizer(data_mgr)

    # Create and run the websocket in a background thread
    ws = build_websocket(data_mgr)
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()

    # Start the Vispy event loop
    viz.run()

if __name__ == "__main__":
    main()
