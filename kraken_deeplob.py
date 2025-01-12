import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WAYLAND_DISPLAY"] = os.environ.get("WAYLAND_DISPLAY", "wayland-0")
os.environ["XDG_SESSION_TYPE"] = "wayland"
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["QT_QPA_PLATFORM"] = "wayland"  # Try wayland explicitly
os.environ["DISPLAY"] = ":0"  # Fallback to X11 if needed
# Basic imports
import sys
import numpy as np
import json
import threading
from collections import deque
from queue import Queue
import logging
import time
import signal

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import vispy
import vispy
from vispy import app, scene, gloo
from vispy.scene.visuals import Text, Sphere, Image
from vispy.util.transforms import translate

# Configure vispy
try:
    app.use_app("egl")
    gloo.gl.use_gl("gl2")
    current_backend = "egl"
except Exception as e:
    print(f"Failed to set EGL backend: {e}")
    current_backend = "unknown"


# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    LSTM,
    Dropout,
    Reshape,
)
from tensorflow.keras.utils import to_categorical

# Import other dependencies
from sklearn.model_selection import train_test_split
import websocket
from dotenv import load_dotenv

# Print system information
print("\nSystem Configuration:")
print(f"Python version: {sys.version}")
print(f"Vispy version: {vispy.__version__}")
print(f"Available backends: {vispy.app.backends.BACKEND_NAMES}")
print(f"Current backend: {current_backend}")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA Available: {tf.test.is_built_with_cuda()}")

# GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory limit to 4GB (4096 MB)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Adjust this number for more/less memory
        )
        print("GPU memory limit set to 4GB")
    except RuntimeError as e:
        print(f"Error setting GPU memory limit: {e}")


def create_deeplob(input_shape):
    try:
        input_layer = Input(name="input", shape=(input_shape))

        # Convolutional layers
        conv1 = Conv2D(32, (1, 2), activation="relu")(input_layer)
        conv2 = Conv2D(32, (1, 2), activation="relu")(conv1)
        conv3 = Conv2D(32, (1, 2), activation="relu")(conv2)
        conv4 = Conv2D(32, (1, 2), activation="relu")(conv3)
        conv5 = Conv2D(32, (1, 2), activation="relu")(conv4)

        # Max pooling layer
        max_pool = MaxPooling2D(pool_size=(1, 2))(conv5)

        # Flatten layer and reshape for LSTM
        flatten = Flatten()(max_pool)
        reshape = Reshape((input_shape[0], -1))(flatten)

        # LSTM layer
        lstm1 = LSTM(64, return_sequences=True)(reshape)
        lstm2 = LSTM(64)(lstm1)

        # Fully connected layers for classification
        dense1 = Dense(128, activation="relu")(lstm2)
        dropout = Dropout(0.5)(dense1)
        output_layer = Dense(3, activation="softmax")(
            dropout
        )  # 3 classes for price movement prediction

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model
    except Exception as e:
        print("Error creating model:", e)
        return None


def prepare_data(orderbook_data, labels, test_size=0.2):
    X = np.array(orderbook_data)
    y = to_categorical(np.array(labels), num_classes=3)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_val, y_train, y_val


def train_deeplob(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    # Add checkpointing
    checkpoint_path = "weights/model_checkpoint"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    # Train with checkpointing
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback],
    )

    return history


def load_model_weights(model, weights_path="weights/model_checkpoint"):
    if os.path.exists(weights_path + ".index"):
        print("Loading saved weights...")
        model.load_weights(weights_path)
        return True
    return False


def make_predictions(model, orderbook_data):
    predictions = model.predict(np.array(orderbook_data))
    return predictions

def check_system_config():
    print("\nSystem Configuration:")
    print(f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE')}")
    print(f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY')}")
    print(f"PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM')}")

    try:
        import subprocess

        nvidia_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ]
        ).decode()
        print(f"NVIDIA GPU: {nvidia_info.strip()}")
    except Exception as e:
        print(f"Could not get NVIDIA info: {e}")

    print("\nVispy Configuration:")
    print(f"Backend: {app.backend_name}")
    print(f"Available backends: {app.backends()}")

###############################################################################
# DataManager: Manages incoming order book data, stores bids/asks, provides geometry
###############################################################################
class DataManager:
    def __init__(self, max_snapshots=100, order_book_depth=200):
        self.max_snapshots = max_snapshots
        self.order_book_depth = order_book_depth
        self.bid_data = {}
        self.ask_data = {}
        self.historical_depth = deque(maxlen=self.max_snapshots)
        self.message_queue = Queue()
        self.last_message_time = 0
        self.previous_predictions = deque(maxlen=100)  # Add this line
        self.prediction_confidence = None  # Add this line
        self.last_update_time = time.time()
        self.min_update_interval = 1.0  # minimum seconds between updates
        self.last_spread = None
        self.min_spread_change_pct = 1.0  # minimum percentage change in spread to trigger update

    def prepare_model_input(self):
        """Prepare order book data for model input"""
        if len(self.historical_depth) < 100:
            return None

        input_data = []
        snapshots = list(self.historical_depth)[-100:]

        for verts, cols in snapshots:
            # Extract bid and ask data from vertices
            bids = {}
            asks = {}

            # Process vertices to reconstruct bid/ask data
            for i in range(0, len(verts), 5):
                x, _, y = verts[i][:3]  # First vertex of rectangle
                volume = y / 10  # Reverse the volume scaling
                price = float(x)

                # Use color to determine if bid or ask
                if cols[i][1] > cols[i][0]:  # Green = bid
                    bids[price] = volume
                else:  # Red = ask
                    asks[price] = volume

            # Format data for model
            snapshot_data = []

            # Get top 20 levels
            bid_prices = sorted(bids.keys(), reverse=True)[:20]
            ask_prices = sorted(asks.keys())[:20]

            # Add bids
            for price in bid_prices:
                snapshot_data.append([price, bids[price]])
            while len(snapshot_data) < 20:  # Pad if needed
                snapshot_data.append([0.0, 0.0])

            # Add asks
            for price in ask_prices:
                snapshot_data.append([price, asks[price]])
            while len(snapshot_data) < 40:  # Pad if needed
                snapshot_data.append([0.0, 0.0])

            input_data.append(snapshot_data)

        return np.array([input_data])  # Shape: (1, 100, 40, 2)


    def save_snapshots_to_file(self, filename):
        """Save the historical snapshots to a file."""
        with open(filename, "w") as f:
            json.dump(list(self.historical_depth), f)
        print(f"Snapshots saved to {filename}")

    def on_message(self, ws, raw_message):
        """Callback from WebSocket, push raw message to queue."""
        try:
            messages = json.loads(raw_message)
            if isinstance(messages, list) and len(messages) > 1:
                current_time = time.time()

                # Process updates
                for b in messages[1].get("b", []):
                    price, volume = float(b[0]), float(b[1])  # Correct for bids
                    if volume == 0:
                        self.bid_data.pop(price, None)
                    elif price > 0:
                        self.bid_data[price] = volume

                for a in messages[1].get("a", []):
                    price, volume = float(a[0]), float(a[1])  # Corrected this line
                    if volume == 0:
                        self.ask_data.pop(price, None)
                    elif price > 0:
                        self.ask_data[price] = volume

                # Check if enough time has passed since last update
                if current_time - self.last_update_time < self.min_update_interval:
                    return

                if self.bid_data and self.ask_data:
                    highest_bid = max(self.bid_data.keys())
                    lowest_ask = min(self.ask_data.keys())
                    current_spread = lowest_ask - highest_bid

                    # Calculate spread change percentage
                    spread_change_pct = 0
                    if self.last_spread:
                        spread_change_pct = abs(current_spread - self.last_spread) / self.last_spread * 100

                    # Update if spread changed significantly
                    if (self.last_spread is None or
                        spread_change_pct >= self.min_spread_change_pct):

                        print(f"\nOrder Book Update ({time.strftime('%H:%M:%S')})")
                        print(f"Highest Bid: {highest_bid:,.1f}")
                        print(f"Lowest Ask:  {lowest_ask:,.1f}")
                        print(f"Spread:      {current_spread:,.1f} ({spread_change_pct:.1f}% change)")

                        self.last_spread = current_spread
                        self.last_update_time = current_time

        except Exception as e:
            print(f"Error processing message: {e}")

    def process_messages(self):
        """Process all pending messages to update bid_data/ask_data."""
        while not self.message_queue.empty():
            raw = self.message_queue.get()
            try:
                messages = json.loads(raw)
                # We expect data like: [channel_id, { "b": [...], "a": [...] }, ...]
                if isinstance(messages, list) and len(messages) > 1:
                    # parse bids
                    for b in messages[1].get("b", []):
                        price, volume = float(b[0]), float(b[1])
                        if volume == 0:
                            self.bid_data.pop(price, None)
                        else:
                            self.bid_data[price] = volume

                    # parse asks
                    for a in messages[1].get("a", []):
                        price, volume = float(a[0]), float(a[1])
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

    def get_market_spread(self):
        """Calculate the spread between the lowest ask and the highest bid."""
        try:
            if not self.bid_data or not self.ask_data:
                return 0

            # Get highest bid and lowest ask
            highest_bid = max(self.bid_data.keys())
            lowest_ask = min(self.ask_data.keys())

            # Validate prices
            if highest_bid > lowest_ask:  # This shouldn't happen in a valid order book
                logger.warning(f"Invalid order book state detected: highest bid ({highest_bid}) > lowest ask ({lowest_ask})")
                return 0

            spread = lowest_ask - highest_bid

            # Additional validation
            if spread < 0:
                logger.warning(f"Negative spread detected: {spread} (bid: {highest_bid}, ask: {lowest_ask})")
                return 0

            return spread

        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return 0

    def record_current_snapshot(self, mid_price):
        """Store a snapshot of the current bid and ask data."""
        # Create deep copies to ensure historical data remains unchanged
        bid_copy = self.bid_data.copy()
        ask_copy = self.ask_data.copy()
        self.historical_depth.append((mid_price, bid_copy, ask_copy))

    def save_snapshots_to_file(self, filename):
        """Save the historical snapshots to a file."""
        with open(filename, "w") as f:
            json.dump(list(self.historical_depth), f)
        print(f"Snapshots saved to {filename}")


###############################################################################
# VaporWaveOrderBookVisualizer: Builds the scene and updates from DataManager
###############################################################################
class VaporWaveOrderBookVisualizer:
    def __init__(self, data_manager, model=None):
        print("Initializing visualizer...")
        self.data = data_manager
        self.model = model
        self.min_samples_for_training = 2048

        try:
            print("Creating canvas with custom configuration...")
            self.canvas = scene.SceneCanvas(
                keys="interactive",
                bgcolor="#220033",
                show=True,
                size=(1024, 768),
                create_native=True,
                vsync=False,
                title="VaporWave Order Book Visualizer",
                dpi=100,
            )

            # Force the canvas to show and update
            self.canvas.show()
            self.canvas.update()

        except Exception as e:
            print(f"Canvas creation failed: {e}")
            raise

        print("Setting up view...")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.distance = 1500
        self.view.camera.center = (0, 0, 0)
        self.view.camera.elevation = 20
        self.view.camera.fov = 60

        # Initialize other variables
        self.camera_z_offset = 0
        self.camera_y_offset = 0
        self.wireframe_type = "rectangle"
        self.last_gpu_check = time.time()
        self.last_prediction_time = time.time()
        self.gpu_check_interval = 5
        self.prediction_interval = 1.0
        self.prediction_history = deque(maxlen=100)
        self.frame_times = deque(maxlen=100)
        self.fps = 0

        print("Initializing scene...")
        self._init_scene()

        # Set up timer
        self.timer = app.Timer(interval=0.1, connect=self.on_timer, start=True)

        print("Setup complete!")


    def train_model(self):
        """train the model on accumulated data"""
        if (
            not self.model
            or len(self.data.historical_depth) < self.min_samples_for_training
        ):
            return
        try:
            # Prepare training data
            X = []
            y = []
            snapshots = list(self.data.historical_depth)

            # Create labeled dataset
            for i in range(100, len(snapshots)):
                # Input features
                input_data = []
                for j in range(i - 100, i):
                    _, bids, asks = snapshots[j]
                    snapshot_data = []

                    # Process bids and asks
                    bid_prices = sorted(bids.keys(), reverse=True)[:20]
                    ask_prices = sorted(asks.keys())[:20]

                    for price in bid_prices:
                        snapshot_data.append([price, bids[price]])
                    for price in ask_prices:
                        snapshot_data.append([price, asks[price]])

                    input_data.append(snapshot_data)

                X.append(input_data)

                # Create label based on price movement
                current_price = snapshots[i][0]
                previous_price = snapshots[i - 1][0]
                if current_price > previous_price:
                    label = 2  # Up
                elif current_price < previous_price:
                    label = 0  # Down
                else:
                    label = 1  # Stable

                y.append(label)

            # Prepare data for training
            X = np.array(X)
            y = to_categorical(np.array(y), num_classes=3)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

            # Train model
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=5,
                batch_size=32,
                verbose=1,
            )

            # Update metrics display
            val_acc = history.history["val_accuracy"][-1]
            self.metrics_label.text = f"Model Accuracy: {val_acc:.2%}"

        except Exception as e:
            print(f"Training error: {e}")

    def make_prediction(self):
        try:
            model_input = self.data.prepare_model_input()
            if model_input is None:
                return None

            prediction = self.model.predict(model_input, verbose=0)
            prediction_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])

            # Log prediction
            direction_map = {0: "DOWN", 1: "STABLE", 2: "UP"}
            print(f"{time.strftime('%H:%M:%S')} | "
                  f"Prediction: {direction_map[prediction_class]} "
                  f"(Confidence: {confidence:.2%})")

            return prediction_class

        except Exception as e:
            print(f"Prediction error: {e}")
            return None


    def visualize_predictions(self, predictions):
        """Visualize model predictions"""
        # Example implementation
        prediction_classes = np.argmax(predictions, axis=1)
        colors = {
            0: (1.0, 0.0, 0.0, 1.0),  # Red for down
            1: (1.0, 1.0, 0.0, 1.0),  # Yellow for neutral
            2: (0.0, 1.0, 0.0, 1.0),  # Green for up
        }

        # Add a prediction indicator or label to the visualization
        prediction_text = ["Down", "Neutral", "Up"][prediction_classes[-1]]
        self.prediction_label.text = f"Prediction: {prediction_text}"
        self.prediction_label.color = colors[prediction_classes[-1]]

        # Add GPU utilization monitoring
        self.last_gpu_check = time.time()
        self.gpu_check_interval = 5  # Check GPU usage every 5 seconds

    def on_timer(self, event):
        try:
            current_time = time.time()


            # Add prediction update
            current_time = time.time()
            if hasattr(self, 'last_prediction_time'):
                if current_time - self.last_prediction_time > self.prediction_interval:
                    if len(self.data.historical_depth) >= 100:
                        self.make_prediction()
                    self.last_prediction_time = current_time


            # Make predictions periodically
            if (
                hasattr(self, "last_prediction_time")
                and current_time - self.last_prediction_time > self.prediction_interval
            ):
                self.make_prediction()
                self.last_prediction_time = current_time

            # GPU usage monitoring
            if (
                hasattr(self, "last_gpu_check")
                and current_time - self.last_gpu_check > self.gpu_check_interval
            ):
                if tf.config.list_physical_devices("GPU"):
                    try:
                        bytes_in_use = tf.config.experimental.get_memory_usage("GPU:0")
                        print(
                            f"GPU Memory in use: {bytes_in_use / (1024 * 1024):.1f} MB"
                        )
                    except Exception as e:
                        print(f"Basic GPU monitoring: Active")
                self.last_gpu_check = current_time

            # Process incoming messages
            self.data.process_messages()
            mid_price = self.data.get_mid_price()
            if mid_price == 0:
                return

            self.current_price_label.text = f"{mid_price:.2f}"
            spread = self.data.get_market_spread()
            self.spread_label.text = f"Spread: {spread:.2f}"
            self.data.record_current_snapshot(mid_price)


        except Exception as e:
            print(f"Error in on_timer: {e}")


    def run(self):
        """Start the Vispy event loop."""
        app.run()


###############################################################################
# WebSocket Handlers
###############################################################################
def on_open(ws):
    print("Authencating with Kraken")
    subscription = {
        "event": "subscribe",
        "pair": ["BTC/USDT"],
        "subscription": {"name": "book", "depth": 1000},
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
        on_message=data_mgr.on_message,
    )

def check_backend_availability():
    """Check which backends are available"""
    print("\nChecking available backends:")
    backends = ["egl", "glfw", "pyglet", "sdl2"]
    for backend in backends:
        try:
            app.use_app(backend)
            print(f"Backend {backend}: Available")
        except Exception as e:
            print(f"Backend {backend}: Not available ({str(e)})")

class OrderBookAnalyzer:
    def __init__(self, data_manager, model=None):
        print("Initializing analyzer...")
        self.data = data_manager
        self.model = model
        self.min_samples_for_training = 1000

        # Initialize tracking variables
        self.prediction_history = deque(maxlen=100)
        self.last_prediction_time = time.time()
        self.prediction_interval = 1.0  # Make prediction every second
        self.last_prediction_confidence = None


        # Initialize GPU monitoring
        self.last_gpu_check = time.time()
        self.gpu_check_interval = 5

        # Set up timer for periodic updates
        self.running = True
        print("Setup complete!")

        def stop(self):
            """Stop the analyzer"""
            print("Stopping analyzer...")
            self.running = False

    def prepare_prediction_input(self):
            """Prepare input data for model prediction"""
            try:
                if len(self.data.historical_depth) < 100:
                    return None

                input_data = []
                snapshots = list(self.data.historical_depth)[-100:]  # Get last 100 snapshots

                for _, bids, asks in snapshots:
                    snapshot_data = []

                    # Process bids
                    bid_prices = sorted(bids.keys(), reverse=True)[:20]
                    for price in bid_prices:
                        snapshot_data.append([float(price), float(bids[price])])

                    # Pad bids if necessary
                    while len(snapshot_data) < 20:
                        snapshot_data.append([0.0, 0.0])

                    # Process asks
                    ask_prices = sorted(asks.keys())[:20]
                    for price in ask_prices:
                        snapshot_data.append([float(price), float(asks[price])])

                    # Pad asks if necessary
                    while len(snapshot_data) < 40:  # Total should be 40 (20 bids + 20 asks)
                        snapshot_data.append([0.0, 0.0])

                    input_data.append(snapshot_data)

                return np.array([input_data])  # Shape: (1, 100, 40, 2)

            except Exception as e:
                print(f"Error preparing prediction input: {e}")
                return None

    def log_trading_info(self, mid_price, spread, prediction=None):
        """Log trading information to console"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # Clear screen for better visibility
        print("\033[H\033[J")  # Clear screen

        print(f"=== Trading Update ({current_time}) ===")

        # Show data collection progress
        snapshots_needed = 100
        current_snapshots = len(self.data.historical_depth)
        if current_snapshots < snapshots_needed:
            progress = (current_snapshots / snapshots_needed) * 100
            print(f"Collecting data: {current_snapshots}/{snapshots_needed} snapshots ({progress:.1f}%)")
            print(f"Time until predictions: ~{(snapshots_needed - current_snapshots) * 5} seconds")
            print()

        print(f"Current Price: ${mid_price:,.2f} ", end="")

        # Calculate and show price change if we have previous data
        if len(self.data.historical_depth) > 1:
            previous_price = list(self.data.historical_depth)[-2][0]
            price_change = mid_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            change_color = "\033[92m" if price_change > 0 else "\033[91m"  # Green/Red
            print(f"{change_color}({price_change:+.2f} | {price_change_pct:+.2f}%)\033[0m")
        else:
            print("(first snapshot)")

        print(f"Spread: ${spread:,.2f} ({(spread/mid_price)*100:.3f}% of price)")

        if prediction is not None:
            prediction_map = {0: "Down ⬇️", 1: "Stable ➡️", 2: "Up ⬆️"}
            confidence = self.last_prediction_confidence if hasattr(self, 'last_prediction_confidence') else None
            conf_str = f" ({confidence:.1%})" if confidence is not None else ""
            print(f"\nPrediction: {prediction_map[prediction]}{conf_str}")
        elif current_snapshots >= snapshots_needed:
            print("\nPrediction: Calculating...")

        # Market depth information
        if self.data.historical_depth:
            latest_snapshot = list(self.data.historical_depth)[-1]
            print("\nMarket Depth:")
            print(f"Bid Levels: {len(latest_snapshot[1])} ({sum(latest_snapshot[1].values()):,.2f} BTC)")
            print(f"Ask Levels: {len(latest_snapshot[2])} ({sum(latest_snapshot[2].values()):,.2f} BTC)")

            # Calculate book imbalance
            total_bids = sum(latest_snapshot[1].values())
            total_asks = sum(latest_snapshot[2].values())
            imbalance = (total_bids - total_asks) / (total_bids + total_asks)
            imbalance_color = "\033[92m" if imbalance > 0 else "\033[91m"
            print(f"Book Imbalance: {imbalance_color}{imbalance:+.2%}\033[0m")

        # Add volatility if we have enough data
        if len(self.data.historical_depth) > 10:
            history_list = list(self.data.historical_depth)
            recent_prices = [price for price, _, _ in history_list[-10:]]
            volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
            print(f"\nVolatility (10 periods): {volatility:.2f}%")

        print("\nPrediction History:")
        if self.prediction_history:
            prediction_map = {0: "⬇️", 1: "➡️", 2: "⬆️"}
            history_str = " ".join(prediction_map[p] for p in list(self.prediction_history)[-10:])
            print(f"Last 10: {history_str}")
        else:
            print("No predictions yet")

        print("=" * 50 + "\n")

    def save_trading_data(self, filename="trading_data.json"):
        """Save trading data to file"""
        try:
            # Convert numpy types to native Python types
            def convert_to_native_types(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return obj

            data = {
                "timestamp": float(time.time()),
                "predictions": [convert_to_native_types(p) for p in list(self.prediction_history)],
                "prices": [(float(time.time()), float(price)) for price, _, _ in self.data.historical_depth],
                "latest_price": float(self.data.get_mid_price()),
                "latest_spread": float(self.data.get_market_spread())
            }

            # Add market stats
            if self.data.historical_depth:
                latest_snapshot = list(self.data.historical_depth)[-1]
                data["market_stats"] = {
                    "bid_levels": int(len(latest_snapshot[1])),
                    "ask_levels": int(len(latest_snapshot[2])),
                    "total_bid_volume": float(sum(latest_snapshot[1].values())),
                    "total_ask_volume": float(sum(latest_snapshot[2].values()))
                }

            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
                print(f"Trading data saved to {filename}")

        except Exception as e:
            print(f"Error saving trading data: {e}")
            import traceback
            traceback.print_exc()

    def make_prediction(self):
        """Make a prediction using the current order book state"""
        if not self.model:
            return None

        try:
            model_input = self.prepare_prediction_input()
            if model_input is None:
                return None

            # Make prediction
            prediction = self.model.predict(model_input, verbose=0)
            prediction_class = np.argmax(prediction[0])

            # Store prediction confidence
            self.last_prediction_confidence = float(np.max(prediction[0]))

            # Store prediction
            self.prediction_history.append(int(prediction_class))

            return prediction_class

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def run(self):
        """Main loop for analysis"""
        last_save_time = time.time()
        save_interval = 60  # Save data every minute
        update_interval = 5  # Update every 5 seconds

        try:
            while self.running:
                current_time = time.time()

                # Make predictions and update display periodically
                if current_time - self.last_prediction_time > update_interval:
                    self.data.process_messages()
                    mid_price = self.data.get_mid_price()

                    if mid_price > 0:
                        spread = self.data.get_market_spread()

                        # Make prediction if we have enough data
                        prediction = None
                        if len(self.data.historical_depth) >= 100:
                            prediction = self.make_prediction()
                            if prediction is not None:
                                # Store prediction confidence
                                model_input = self.prepare_prediction_input()
                                if model_input is not None:
                                    pred_probabilities = self.model.predict(model_input, verbose=0)
                                    self.last_prediction_confidence = np.max(pred_probabilities[0])

                        # Log information
                        self.log_trading_info(mid_price, spread, prediction)

                        # Save data periodically
                        if current_time - last_save_time > save_interval:
                            filename = f"kraken_data/trading_data_{int(current_time)}.json"
                            self.save_trading_data(filename)
                            last_save_time = current_time

                        # Record snapshot
                        self.data.record_current_snapshot(mid_price)

                    self.last_prediction_time = current_time

                # Monitor GPU usage
                if current_time - self.last_gpu_check > self.gpu_check_interval:
                    if tf.config.list_physical_devices('GPU'):
                        try:
                            bytes_in_use = tf.config.experimental.get_memory_usage('GPU:0')
                            print(f"GPU Memory in use: {bytes_in_use / (1024 * 1024):.1f} MB")
                        except Exception:
                            print("Basic GPU monitoring: Active")
                    self.last_gpu_check = current_time

                # Small sleep to prevent CPU overuse
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping analysis gracefully...")
        except Exception as e:
            print(f"Error in analysis loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                final_filename = f"kraken_data/final_trading_data_{int(time.time())}.json"
                self.save_trading_data(final_filename)
                print(f"Final data saved to {final_filename}")
            except Exception as e:
                print(f"Error saving final data: {e}")


###############################################################################
# Main Execution
###############################################################################


def main():
    load_dotenv()

    # Create directories
    os.makedirs("kraken_data", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    print("\nInitializing Neural Network Trading System...")

    # Initialize model
    print("Creating model...")
    input_shape = (100, 40, 2)
    model = create_deeplob(input_shape)
    load_model_weights(model)

    print("Setting up data manager...")
    data_mgr = DataManager(max_snapshots=200, order_book_depth=500)

    print("Creating analyzer...")
    analyzer = OrderBookAnalyzer(data_mgr, model=model)

    print("Setting up WebSocket...")
    ws = build_websocket(data_mgr)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        try:
            if hasattr(analyzer, 'running'):
                analyzer.running = False
            ws.close()
            data_mgr.save_snapshots_to_file("kraken_data/final_snapshot.json")
        except Exception as e:
            print(f"Error during shutdown: {e}")

    signal.signal(signal.SIGINT, signal_handler)
    print("\nStarting system...")

    # Start WebSocket in a separate thread
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

    # Run the analyzer
    analyzer.run()

if __name__ == "__main__":
    main()
