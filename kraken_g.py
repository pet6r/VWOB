import os
import sys
import json
import time
import signal
import threading
import logging
from collections import deque
from queue import Queue

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, Reshape
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import websocket
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
console = Console()

# configure GPU
def configure_gpu(memory_limit=4096):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )
            logger.info(f"GPU memory limit set to {memory_limit}MB")
        except RuntimeError as e:
            logger.error(f"Error configuring GPU: {e}")

# model creation
def create_deeplob(input_shape):
    input_layer = Input(shape=input_shape)
    x = input_layer
    for _ in range(5):
        x = Conv2D(32, (1, 2), activation="relu")(x)
    x = MaxPooling2D((1, 2))(x)
    x = Flatten()(x)
    x = Reshape((input_shape[0], -1))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(3, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# data preparation
def prepare_data(orderbook_data, labels, test_size=0.2):
    X = np.array(orderbook_data)
    y = to_categorical(np.array(labels), num_classes=3)
    return train_test_split(X, y, test_size=test_size, random_state=42)

# helper functions
def load_weights(model, path="weights/model_checkpoint"):
    if os.path.exists(f"{path}.index"):
        logger.info("Loading saved weights...")
        model.load_weights(path)
        return True
    logger.info("No pre-trained weights found.")
    return False

def save_to_file(filename, data):
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

# websocket logic
class WebSocketHandler:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def on_open(self, ws):
        logger.info("WebSocket connected")
        subscription = {
            "event": "subscribe",
            "pair": ["BTC/USDT"],
            "subscription": {"name": "book", "depth": 1000}
        }
        ws.send(json.dumps(subscription))
        logger.info("Subscribed to order book")

    def on_message(self, ws, message):
        self.data_manager.process_message(json.loads(message))

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, status_code, msg):
        logger.info(f"WebSocket closed: {status_code} - {msg}")

# data manager
class DataManager:
    def __init__(self, max_snapshots=200, snapshot_interval=10, min_samples=100):
        self.bid_data = {}
        self.ask_data = {}
        self.historical_depth = deque(maxlen=max_snapshots)
        self.last_snapshot_time = time.time()
        self.snapshot_interval = snapshot_interval
        self.min_samples = min_samples
        self.last_logged_snapshot = None

    def process_message(self, message):
        try:
            if not isinstance(message, list) or len(message) <= 1:
                return

            for b in message[1].get("b", []):
                self._update_order(self.bid_data, b)

            for a in message[1].get("a", []):
                self._update_order(self.ask_data, a)

            if time.time() - self.last_snapshot_time >= self.snapshot_interval:
                self.record_snapshot()
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _update_order(self, order_dict, order):
        price, volume = float(order[0]), float(order[1])
        if volume == 0:
            order_dict.pop(price, None)
        else:
            order_dict[price] = volume

    def record_snapshot(self):
        if self.bid_data and self.ask_data:
            highest_bid = max(self.bid_data, default=0)
            lowest_ask = min(self.ask_data, default=float('inf'))
            mid_price = (highest_bid + lowest_ask) / 2
            self.historical_depth.append((mid_price, self.bid_data.copy(), self.ask_data.copy()))
            self.last_snapshot_time = time.time()

            if mid_price != self.last_logged_snapshot:
                console.log(f"[bold yellow]Snapshot recorded: Mid Price = {mid_price}")
                self.last_logged_snapshot = mid_price

    def get_progress_info(self):
        current_samples = len(self.historical_depth)
        progress = (current_samples / self.min_samples) * 100
        remaining_samples = self.min_samples - current_samples
        time_remaining = remaining_samples * self.snapshot_interval
        minutes, seconds = divmod(time_remaining, 60)
        return progress, minutes, seconds

# main logic
def main():
    load_dotenv()
    configure_gpu()

    model = create_deeplob((100, 40, 2))
    load_weights(model)

    data_manager = DataManager()
    ws_handler = WebSocketHandler(data_manager)

    ws = websocket.WebSocketApp(
        "wss://ws.kraken.com/",
        on_open=ws_handler.on_open,
        on_message=ws_handler.on_message,
        on_error=ws_handler.on_error,
        on_close=ws_handler.on_close
    )

    shutdown_event = threading.Event()

    def shutdown(*_):
        if not shutdown_event.is_set():
            logger.info("Shutting down...")
            shutdown_event.set()
            save_to_file("final_snapshot.json", list(data_manager.historical_depth))
            ws.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Collecting data for predictions...", total=100)

        while not shutdown_event.is_set():
            progress_percentage, minutes, seconds = data_manager.get_progress_info()
            progress.update(task, completed=progress_percentage)

            if int(progress_percentage) % 10 == 0 and progress_percentage > 0:
                console.log(f"[bold cyan]Time until predictions: {int(minutes)}m {int(seconds)}s")

            if progress_percentage >= 100:
                console.log("[bold green]Enough data collected! Ready for predictions.")
                break

            time.sleep(1)

if __name__ == "__main__":
    main()
