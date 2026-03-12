import argparse
import atexit
import json
import threading
from collections import deque
from queue import Queue

import numpy as np
import websocket
from vispy import app, scene
from vispy.scene.visuals import Text, Line, Markers


def fmt_quote(val):
    if val >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    elif val >= 1_000:
        return f"${val / 1_000:.1f}K"
    return f"${val:.0f}"


###############################################################################
# DataManager: WebSocket ingestion, snapshot history, and mesh construction
###############################################################################
class DataManager:
    def __init__(self, base_asset="BTC", max_snapshots=250):
        self.base_asset = base_asset
        self.max_snapshots = max_snapshots
        self.view_depth = 95  # how many levels deep to render

        self.bid_data = {}
        self.ask_data = {}
        self.historical_depth = deque(maxlen=self.max_snapshots)

        self._dirty = False
        self.previous_mid_price = None

        self.message_queue = Queue()

    def on_message(self, _ws, msg):
        self.message_queue.put(msg)

    def process_messages(self):
        had_data = False
        while not self.message_queue.empty():
            raw = self.message_queue.get()
            try:
                data = json.loads(raw)
                if isinstance(data, list) and len(data) > 1:
                    for b in data[1].get("b", []):
                        price, volume = float(b[0]), float(b[1])
                        if volume == 0:
                            self.bid_data.pop(price, None)
                        else:
                            self.bid_data[price] = volume
                        had_data = True

                    for a in data[1].get("a", []):
                        price, volume = float(a[0]), float(a[1])
                        if volume == 0:
                            self.ask_data.pop(price, None)
                        else:
                            self.ask_data[price] = volume
                        had_data = True
            except Exception as e:
                print("Error processing message:", e)
        if had_data:
            self._dirty = True

    def get_mid_price(self):
        if self.bid_data and self.ask_data:
            highest_bid = max(self.bid_data.keys())
            lowest_ask = min(self.ask_data.keys())
            return (highest_bid + lowest_ask) / 2
        return 0

    def has_changed(self):
        if self._dirty:
            self._dirty = False
            return True
        return False

    def record_current_snapshot(self, mid_price):
        self.previous_mid_price = mid_price
        self.historical_depth.append((mid_price, dict(self.bid_data), dict(self.ask_data)))

    def build_mountain_mesh(self, price_resolution=100):
        """Build wireframe mountain mesh from historical order book snapshots.

        Each time-slice is a cumulative volume profile centered on its own
        mid-price, so older frames stay anchored even as price moves.
        
        Returns (vertices, colors, line_connect, faces).
        """
        empty4 = (np.zeros((0, 3)), np.zeros((0, 4)), np.zeros((0, 2), dtype=np.uint32), np.zeros((0, 3), dtype=np.uint32))
        if len(self.historical_depth) < 2:
            return empty4

        n_time = len(self.historical_depth)
        n_price = price_resolution
        depth = self.view_depth

        best_bids = [max(bids.keys()) if bids else None for _, bids, _ in self.historical_depth]
        best_asks = [min(asks.keys()) if asks else None for _, _, asks in self.historical_depth]
        mid_prices = [(b + a) / 2 for b, a in zip(best_bids, best_asks) if b is not None and a is not None]
        if not mid_prices:
            return empty4

        current_mid = mid_prices[-1]

        # Price span for the x-axis.  Uses a rolling max over recent
        # snapshots so the width the visual depth doesn't jitter on every tick. The minimum
        # span is percentage-based (1% of mid) so it adapts to the asset aka DOGE vs BTC
        min_span = max(current_mid * 0.01, 1e-8)
        window = min(30, n_time)
        recent = list(self.historical_depth)[-window:]
        snap_spans = []
        for _, sbids, sasks in recent:
            lb = sorted(sbids.keys(), reverse=True)[:depth]
            la = sorted(sasks.keys())[:depth]
            bs = (current_mid - lb[-1]) if lb else min_span
            as_ = (la[-1] - current_mid) if la else min_span
            snap_spans.append(max(bs, as_, min_span))
        span = max(snap_spans) * 1.15 if snap_spans else min_span

        time_spacing = 15
        time_axis = (n_time - 1 - np.arange(n_time)) * time_spacing

        total = n_time * n_price
        half_price = n_price // 2
        verts = np.empty((total, 3), dtype=np.float32)
        cols = np.empty((total, 4), dtype=np.float32)

        x_half_width = 600

        for ti, (mid_price, bids, asks) in enumerate(self.historical_depth):
            sorted_bids = sorted(bids.items(), reverse=True)[:depth]
            sorted_asks = sorted(asks.items())[:depth]

            price_axis = np.linspace(mid_price - span, mid_price + span, n_price)
            x_coords = (price_axis - mid_price) / span * x_half_width

            # Bids sorted ascending so cumsum rises toward mid (mountain shape)
            if sorted_bids:
                bid_asc = sorted(sorted_bids)
                bp = np.array([p for p, _ in bid_asc])
                bv = np.cumsum([v for _, v in bid_asc])
            else:
                bp, bv = np.array([]), np.array([])

            if sorted_asks:
                ap = np.array([p for p, _ in sorted_asks])
                av = np.cumsum([v for _, v in sorted_asks])
            else:
                ap, av = np.array([]), np.array([])

            offset = ti * n_price
            y_val = time_axis[ti]

            for pi in range(n_price):
                price = price_axis[pi]
                idx = offset + pi
                if price <= mid_price:
                    # Cum bid volume at this price = total bid vol
                    # minus cumsum of bids below this price
                    below = bp < price
                    if below.any():
                        vol = bv[-1] - bv[below][-1]
                    elif len(bv):
                        vol = bv[-1]
                    else:
                        vol = 0
                    cols[idx] = [0.0, 0.85, 0.75, 1.0]
                else:
                    mask = ap <= price
                    vol = av[mask][-1] if mask.any() else 0
                    cols[idx] = [0.95, 0.1, 0.75, 1.0]

                verts[idx] = [x_coords[pi], y_val, vol]

        # Shift each time-slice so the valley (spread) sits at z=0
        raw_z = verts[:, 2].copy().reshape(n_time, n_price)
        for ti in range(n_time):
            raw_z[ti] -= raw_z[ti].min()

        # Normalize bid and ask sides independently so an asymmetric
        # book doesn't squash the smaller side to the floor
        target_height = x_half_width * 0.7
        bid_z = raw_z[:, :half_price]
        ask_z = raw_z[:, half_price:]

        bid_max = bid_z.max() if bid_z.max() > 0 else 1
        ask_max = ask_z.max() if ask_z.max() > 0 else 1

        bid_z = np.sqrt(bid_z) / np.sqrt(bid_max) * target_height
        ask_z = np.sqrt(ask_z) / np.sqrt(ask_max) * target_height

        verts[:, 2] = np.concatenate([bid_z, ask_z], axis=1).ravel()

        # Build wireframe connectivity (horizontal + vertical segments)
        idx_grid = np.arange(total).reshape(n_time, n_price)
        h_left = idx_grid[:, :-1].ravel()
        h_right = idx_grid[:, 1:].ravel()
        v_top = idx_grid[:-1, :].ravel()
        v_bot = idx_grid[1:, :].ravel()

        connect = np.concatenate([
            np.column_stack([h_left, h_right]),
            np.column_stack([v_top, v_bot]),
        ]).astype(np.uint32)

        # Build triangle faces for solid surface fill
        tl = idx_grid[:-1, :-1].ravel()  # top-left
        tr = idx_grid[:-1, 1:].ravel()   # top-right
        bl = idx_grid[1:, :-1].ravel()   # bottom-left
        br = idx_grid[1:, 1:].ravel()    # bottom-right
        faces = np.concatenate([
            np.column_stack([tl, tr, br]),
            np.column_stack([tl, br, bl]),
        ]).astype(np.uint32)

        return verts, cols, connect, faces

    def get_market_control(self):
        total_bid = sum(self.bid_data.values())
        total_ask = sum(self.ask_data.values())
        if total_bid > total_ask:
            return "Buyers in Control"
        elif total_ask > total_bid:
            return "Sellers in Control"
        return "Neutral"

    def get_book_stats(self):
        """Return cumulative totals and largest single wall on each side."""
        bid_total_base = sum(self.bid_data.values())
        ask_total_base = sum(self.ask_data.values())
        bid_total_quote = sum(p * v for p, v in self.bid_data.items())
        ask_total_quote = sum(p * v for p, v in self.ask_data.items())

        bid_cliff_price, bid_cliff_vol = 0, 0
        if self.bid_data:
            bid_cliff_price = max(self.bid_data, key=self.bid_data.get)
            bid_cliff_vol = self.bid_data[bid_cliff_price]

        ask_cliff_price, ask_cliff_vol = 0, 0
        if self.ask_data:
            ask_cliff_price = max(self.ask_data, key=self.ask_data.get)
            ask_cliff_vol = self.ask_data[ask_cliff_price]

        return {
            "bid_total_base": bid_total_base,
            "ask_total_base": ask_total_base,
            "bid_total_quote": bid_total_quote,
            "ask_total_quote": ask_total_quote,
            "bid_cliff_price": bid_cliff_price,
            "bid_cliff_vol": bid_cliff_vol,
            "bid_cliff_quote": bid_cliff_price * bid_cliff_vol,
            "ask_cliff_price": ask_cliff_price,
            "ask_cliff_vol": ask_cliff_vol,
            "ask_cliff_quote": ask_cliff_price * ask_cliff_vol,
        }


###############################################################################
# MountainVisualizer: Builds the scene and updates from DataManager
###############################################################################
class MountainVisualizer:
    def __init__(self, data_manager):
        self.data = data_manager

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor="#0a0a1a",
            show=True,
            config={"depth_size": 24, "samples": 8},
        )
        self.view = self.canvas.central_widget.add_view()

        self.view.camera = "turntable"
        self.view.camera.distance = 1800
        self.view.camera.center = (0, 0, 0)
        self.view.camera.elevation = 35
        self.view.camera.azimuth = 20
        self.view.camera.fov = 45

        self._init_scene()
        self.timer = app.Timer(interval=0.1, connect=self.on_timer, start=True)

        self._stats_visible = True
        self._fly_speed = 80
        self.canvas.events.key_press.connect(self._on_key_press)

    def _on_key_press(self, event):
        cam = self.view.camera
        s = self._fly_speed
        if event.key == 'Left':
            cam.azimuth -= 5
        elif event.key == 'Right':
            cam.azimuth += 5
        elif event.key == 'Up':
            cam.elevation = min(cam.elevation + 5, 89)
        elif event.key == 'Down':
            cam.elevation = max(cam.elevation - 5, -89)
        elif event.key in ('+', '='):
            cam.distance = max(cam.distance - s * 3, 100)
        elif event.key in ('-', '_'):
            cam.distance += s * 3
        elif event.key == 'W':
            az = np.radians(cam.azimuth)
            cx, cy, cz = cam.center
            cam.center = (cx + s * np.sin(az), cy + s * np.cos(az), cz)
        elif event.key == 'S':
            az = np.radians(cam.azimuth)
            cx, cy, cz = cam.center
            cam.center = (cx - s * np.sin(az), cy - s * np.cos(az), cz)
        elif event.key == 'A':
            az = np.radians(cam.azimuth - 90)
            cx, cy, cz = cam.center
            cam.center = (cx + s * np.sin(az), cy + s * np.cos(az), cz)
        elif event.key == 'D':
            az = np.radians(cam.azimuth + 90)
            cx, cy, cz = cam.center
            cam.center = (cx + s * np.sin(az), cy + s * np.cos(az), cz)
        elif event.key == 'R':
            cam.distance = 1800
            cam.center = (0, 0, 0)
            cam.elevation = 35
            cam.azimuth = 20
        elif event.key == '[':
            self.data.view_depth = max(5, self.data.view_depth - 5)
            self.depth_label.text = f"DEPTH: {self.data.view_depth} levels"
            self.data._dirty = True  # force rebuild
        elif event.key == ']':
            self.data.view_depth = min(1000, self.data.view_depth + 5)
            self.depth_label.text = f"DEPTH: {self.data.view_depth} levels"
            self.data._dirty = True  # force rebuild
        elif event.key == 'I':
            self._stats_visible = not self._stats_visible
            for el in (self.hud_bg, self.stats_header, self.bid_total_label,
                       self.ask_total_label, self.stats_divider,
                       self.bid_cliff_label, self.ask_cliff_label,
                       self.depth_label):
                el.visible = self._stats_visible
        self.canvas.update()

    def _build_sun_disc(self, radius, n_rings, sun_y):
        """Flat disc mesh with orange→magenta vertical gradient."""
        verts = [[0.0, sun_y, 0.0]]  # center vertex
        cols = [[1.0, 0.6, 0.1, 1.0]]
        n_seg = 40

        for ri in range(1, n_rings + 1):
            r = radius * ri / n_rings
            for si in range(n_seg):
                angle = 2.0 * np.pi * si / n_seg
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                verts.append([x, sun_y, z])
                t = (z + radius) / (2.0 * radius)  # 0=bottom, 1=top
                cr = 0.8 + 0.2 * t
                cg = 0.05 + 0.65 * t
                cb = 0.4 - 0.3 * t
                cols.append([cr, cg, cb, 1.0])

        verts = np.array(verts, dtype=np.float32)
        cols = np.array(cols, dtype=np.float32)

        faces = []
        # Center fan
        for si in range(n_seg):
            faces.append([0, 1 + si, 1 + (si + 1) % n_seg])
        # Ring strips
        for ri in range(1, n_rings):
            off0 = 1 + (ri - 1) * n_seg
            off1 = 1 + ri * n_seg
            for si in range(n_seg):
                s1 = (si + 1) % n_seg
                faces.append([off0 + si, off1 + si, off1 + s1])
                faces.append([off0 + si, off1 + s1, off0 + s1])

        faces = np.array(faces, dtype=np.uint32)
        return verts, cols, faces

    def _init_scene(self):
        sun_y = 4800
        sun_r = 1000

        # vaporwave purple backdrop/horizon → deep indigo top
        sky_size = 8000
        sky_verts = np.array([
            [-sky_size, sun_y + 200, -sky_size * 0.3],
            [ sky_size, sun_y + 200, -sky_size * 0.3],
            [ sky_size, sun_y + 200,  sky_size],
            [-sky_size, sun_y + 200,  sky_size],
        ], dtype=np.float32)
        sky_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        sky_colors = np.array([
            [0.15, 0.02, 0.18, 1.0],
            [0.15, 0.02, 0.18, 1.0],
            [0.02, 0.01, 0.06, 1.0],
            [0.02, 0.01, 0.06, 1.0],
        ], dtype=np.float32)
        self.sky = scene.visuals.Mesh(
            vertices=sky_verts, faces=sky_faces,
            vertex_colors=sky_colors, parent=self.view.scene,
        )
        self.sky.set_gl_state(depth_test=True)

        # Stars with no overlap on the sunset
        n_stars = 250
        star_x = np.random.uniform(-6000, 6000, n_stars)
        star_y = np.full(n_stars, sun_y + 150)
        star_z = np.random.uniform(200, 5000, n_stars)
        dist_sq = star_x ** 2 + star_z ** 2
        outside_sun = dist_sq > (sun_r * 1.15) ** 2
        star_x = star_x[outside_sun]
        star_y = star_y[outside_sun]
        star_z = star_z[outside_sun]
        n_kept = len(star_x)
        star_pos = np.column_stack([star_x, star_y, star_z]).astype(np.float32)
        star_sizes = np.random.uniform(2, 8, n_kept).astype(np.float32)
        star_alpha = np.random.uniform(0.3, 1.0, n_kept)
        star_colors = np.column_stack([
            np.full(n_kept, 0.9),
            np.full(n_kept, 0.85),
            np.full(n_kept, 1.0),
            star_alpha,
        ]).astype(np.float32)
        self.stars = Markers(parent=self.view.scene)
        self.stars.set_data(
            pos=star_pos, size=star_sizes, face_color=star_colors,
            edge_width=0,
        )
        self.stars.set_gl_state(
            blend=True, depth_test=True,
            blend_func=('src_alpha', 'one'),
        )

        # Sun disc
        sun_verts, sun_cols, sun_faces = self._build_sun_disc(sun_r, 20, sun_y)
        self.sun = scene.visuals.Mesh(
            vertices=sun_verts, faces=sun_faces,
            vertex_colors=sun_cols, parent=self.view.scene,
        )
        self.sun.set_gl_state(depth_test=True)

        # Sun bands (retro horizontal stripes, wider toward bottom)
        # TO DO: fix the sun bands, they are not visible

        band_centers = [-300, -180, -80, 10, 90, 165, 235]
        band_heights = [80, 55, 40, 30, 22, 16, 12]
        band_verts_all = []
        band_faces_all = []
        band_cols_all = []
        band_y = sun_y - 1
        bg_color = [0.04, 0.01, 0.06, 1.0]

        for cz, h in zip(band_centers, band_heights):
            base = len(band_verts_all)
            hw = sun_r * 1.05
            band_verts_all.extend([
                [-hw, band_y, cz - h / 2],
                [ hw, band_y, cz - h / 2],
                [ hw, band_y, cz + h / 2],
                [-hw, band_y, cz + h / 2],
            ])
            band_faces_all.extend([
                [base, base + 1, base + 2],
                [base, base + 2, base + 3],
            ])
            band_cols_all.extend([bg_color] * 4)

        band_verts_all = np.array(band_verts_all, dtype=np.float32)
        band_faces_all = np.array(band_faces_all, dtype=np.uint32)
        band_cols_all = np.array(band_cols_all, dtype=np.float32)
        self.sun_bands = scene.visuals.Mesh(
            vertices=band_verts_all, faces=band_faces_all,
            vertex_colors=band_cols_all, parent=self.view.scene,
        )
        self.sun_bands.set_gl_state(depth_test=True)

        # Floor plane
        floor_size = 6000
        floor_verts = np.array([
            [-floor_size, -floor_size, 0],
            [ floor_size, -floor_size, 0],
            [ floor_size,  floor_size, 0],
            [-floor_size,  floor_size, 0],
        ], dtype=np.float32)
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        floor_colors = np.array([
            [0.06, 0.01, 0.10, 0.95],
            [0.06, 0.01, 0.10, 0.95],
            [0.10, 0.02, 0.15, 0.80],
            [0.10, 0.02, 0.15, 0.80],
        ], dtype=np.float32)
        self.floor = scene.visuals.Mesh(
            vertices=floor_verts, faces=floor_faces,
            vertex_colors=floor_colors, parent=self.view.scene,
        )
        self.floor.set_gl_state(
            blend=True, depth_test=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

        # Neon grid
        self.grid = scene.visuals.GridLines(
            color=(1.0, 0.2, 0.8, 0.6), parent=self.view.scene
        )
        self.grid.transform = scene.transforms.STTransform(
            scale=(3, 1, 3), translate=(0, 0, 1)
        )

        # Midprice road (white center line along the valley/trough)
        road_pts = np.array([
            [0, -100, 1],
            [0, 5000, 1],
        ], dtype=np.float32)
        self.midprice_road = Line(
            pos=road_pts,
            color=(1.0, 1.0, 1.0, 0.25),
            width=1.5,
            parent=self.view.scene,
        )
        self.midprice_road.set_gl_state(
            blend=True, depth_test=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'),
        )

        # Data visuals
        self.mountain_mesh = scene.visuals.Mesh(parent=self.view.scene)
        self.mountain_mesh.set_gl_state(
            blend=True, depth_test=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'),
        )

        self.mountain_line = Line(parent=self.view.scene, width=1.5)
        self.mountain_line.set_gl_state(depth_test=True)

        self.ghost_line = Line(parent=self.view.scene, width=1.0)  # faint glow layer
        self.ghost_line.set_gl_state(
            blend=True, depth_test=True, blend_func=("src_alpha", "one"), line_width=1.0
        )

        # HUD panel
        panel_w, panel_h = 270, 220
        self.hud_bg = scene.visuals.Rectangle(
            center=(panel_w / 2, panel_h / 2),
            width=panel_w, height=panel_h,
            color=(0.0, 0.0, 0.08, 0.72),
            border_color=(0.4, 0.2, 0.6, 0.6),
            border_width=0,
            parent=self.canvas.scene,
        )
        self.hud_bg.set_gl_state(blend=True, depth_test=False,
                                  blend_func=('src_alpha', 'one_minus_src_alpha'))

        # Price label
        self.current_price_label = Text(
            text="?",
            color=(1.0, 1.0, 1.0, 1.0),
            font_size=20,
            bold=True,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.current_price_label.transform = scene.transforms.STTransform(
            translate=(8, 8)
        )

        self.control_label = Text(
            text="Neutral",
            color=(0.85, 0.75, 1.0, 1.0),
            font_size=10,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.control_label.transform = scene.transforms.STTransform(
            translate=(12, 46)
        )

        # Orderbook stats
        self.stats_header = Text(
            text="ORDERBOOK",
            color=(0.7, 0.5, 1.0, 1.0),
            font_size=8,
            bold=True,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.stats_header.transform = scene.transforms.STTransform(
            translate=(10, 72)
        )

        self.bid_total_label = Text(
            text="",
            color=(0.1, 1.0, 0.75, 1.0),
            font_size=9,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.bid_total_label.transform = scene.transforms.STTransform(
            translate=(12, 90)
        )

        self.ask_total_label = Text(
            text="",
            color=(1.0, 0.25, 0.65, 1.0),
            font_size=9,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.ask_total_label.transform = scene.transforms.STTransform(
            translate=(12, 108)
        )

        self.stats_divider = Text(
            text="LARGEST WALLS",
            color=(0.7, 0.5, 1.0, 1.0),
            font_size=8,
            bold=True,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.stats_divider.transform = scene.transforms.STTransform(
            translate=(10, 126)
        )

        self.bid_cliff_label = Text(
            text="",
            color=(0.1, 1.0, 0.75, 1.0),
            font_size=9,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.bid_cliff_label.transform = scene.transforms.STTransform(
            translate=(12, 144)
        )

        self.ask_cliff_label = Text(
            text="",
            color=(1.0, 0.25, 0.65, 1.0),
            font_size=9,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.ask_cliff_label.transform = scene.transforms.STTransform(
            translate=(12, 162)
        )

        # Depth level indicator
        self.depth_label = Text(
            text=f"VISUAL DEPTH: {self.data.view_depth} levels",
            color=(0.5, 0.85, 1.0, 1.0),
            font_size=9,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.depth_label.transform = scene.transforms.STTransform(
            translate=(12, 180)
        )

        # Controls help text
        self.key_help_label = Text(
            text=(
                "Arrow Keys: Look  |  WASD: Fly  |  +/-: Zoom  |  [/]: Depth Levels  |  R: Reset"
            ),
            color=(1.0, 1.0, 1.0, 0.3),
            font_size=8,
            anchor_x="right",
            anchor_y="top",
            parent=self.canvas.scene,
        )
        self._update_key_help_pos()
        self.canvas.events.resize.connect(self._on_resize)

    def _update_key_help_pos(self):
        w, h = self.canvas.size
        self.key_help_label.transform = scene.transforms.STTransform(
            translate=(w - 10, h - 8)
        )

    def _on_resize(self, _event):
        self._update_key_help_pos()

    def on_timer(self, _event):
        self.data.process_messages()

        if not self.data.has_changed():
            return

        mid_price = self.data.get_mid_price()
        self.data.record_current_snapshot(mid_price)

        verts, cols, connect, faces = self.data.build_mountain_mesh(price_resolution=120)

        if len(verts) == 0 or len(connect) == 0:
            self.canvas.update()
            return

        # Solid surface
        mesh_cols = cols.copy()
        mesh_cols[:, :3] *= 0.5
        mesh_cols[:, 3] = 0.7
        self.mountain_mesh.set_data(
            vertices=verts, faces=faces, vertex_colors=mesh_cols,
        )

        self.mountain_line.set_data(pos=verts, color=cols, connect=connect)

        ghost_cols = cols.copy()
        ghost_cols[:, 3] *= 0.3  # transparent glow offset
        self.ghost_line.set_data(pos=verts * 1.003, color=ghost_cols, connect=connect)

        self.current_price_label.text = f"{float(mid_price):.2f}"
        self.control_label.text = self.data.get_market_control()

        stats = self.data.get_book_stats()

        base_sym = self.data.base_asset

        self.bid_total_label.text = (
            f"| BID  {stats['bid_total_base']:>8.2f} {base_sym}  {fmt_quote(stats['bid_total_quote']):>8s}"
        )
        self.ask_total_label.text = (
            f"| ASK  {stats['ask_total_base']:>8.2f} {base_sym}  {fmt_quote(stats['ask_total_quote']):>8s}"
        )

        self.bid_cliff_label.text = (
            f"  {stats['bid_cliff_vol']:.4f} {base_sym} @ {stats['bid_cliff_price']:.2f}"
            f"  {fmt_quote(stats['bid_cliff_quote'])}"
        )
        self.ask_cliff_label.text = (
            f"  {stats['ask_cliff_vol']:.4f} {base_sym} @ {stats['ask_cliff_price']:.2f}"
            f"  {fmt_quote(stats['ask_cliff_quote'])}"
        )

        self.canvas.update()

    def run(self):
        app.run()


###############################################################################
# WebSocket 
###############################################################################
def build_websocket(data_mgr, pair):
    def on_open(ws):
        print(f"Subscribing to {pair}")
        ws.send(json.dumps({
            "event": "subscribe",
            "pair": [pair],
            "subscription": {"name": "book", "depth": 1000},
        }))
        print(f"Subscribed to {pair} order book")

    return websocket.WebSocketApp(
        "wss://ws.kraken.com/",
        on_open=on_open,
        on_error=lambda _ws, e: print("WebSocket error:", e),
        on_close=lambda _ws, c, m: print("WebSocket closed:", c, m),
        on_message=data_mgr.on_message,
    )


def main():
    parser = argparse.ArgumentParser(description="Kraken Orderbook Mountain Visualizer")
    parser.add_argument("--pair", type=str, default="XBT/USD", help="Trading pair to visualize (e.g., XBT/USDT, ETH/USD)")
    args = parser.parse_args()

    pair = args.pair
    base_asset = pair.split("/")[0] if "/" in pair else pair

    data_mgr = DataManager(base_asset=base_asset)

    viz = MountainVisualizer(data_mgr)

    ws = build_websocket(data_mgr, pair)
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()

    atexit.register(ws.close)

    viz.run()

if __name__ == "__main__":
    main()
