import os
import sys
import numpy as np
import json
import threading
from collections import deque
from queue import Queue
import logging
import time
import signal

# Suppress TensorFlow warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
        print("GPU memory limit set to 4GB")
    except RuntimeError as e:
        print(f"Error setting GPU memory limit: {e}")

def create_deeplob(input_shape):
    """Create the DeepLOB model architecture"""
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

        # LSTM layers
        lstm1 = LSTM(64, return_sequences=True)(reshape)
        lstm2 = LSTM(64)(lstm1)

        # Fully connected layers
        dense1 = Dense(128, activation="relu")(lstm2)
        dropout = Dropout(0.5)(dense1)
        output_layer = Dense(3, activation="softmax")(dropout)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return None

def load_model_weights(model, weights_path="weights/model_checkpoint"):
    """Load pre-trained weights if available"""
    if os.path.exists(weights_path + ".index"):
        logger.info("Loading saved weights...")
        model.load_weights(weights_path)
        return True
    logger.info("No pre-trained weights found.")
    return False

def prepare_data(orderbook_data, labels, test_size=0.2):
    """Prepare data for training"""
    X = np.array(orderbook_data)
    y = to_categorical(np.array(labels), num_classes=3)
    return train_test_split(X, y, test_size=test_size, random_state=42)


class DataManager:
    def __init__(self, max_snapshots=200, order_book_depth=500, max_retries=3, retry_delay=0.1):
        self.max_snapshots = max_snapshots
        self.order_book_depth = order_book_depth
        self.bid_data = {}
        self.ask_data = {}
        self.historical_depth = deque(maxlen=self.max_snapshots)
        self.message_queue = Queue()
        self.last_message_time = time.time()
        self.last_update_time = time.time()
        self.previous_predictions = deque(maxlen=100)
        self.prediction_confidence = None
        self.snapshot_interval = 10.0  # Take snapshot every 10 seconds
        self.last_snapshot_time = time.time()
        self.snapshot_count = 0
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def should_take_snapshot(self):
        """Determine if we should take a new snapshot"""
        current_time = time.time()
        time_since_last = current_time - self.last_snapshot_time

        # Only take snapshot if 10 seconds have passed
        return (time_since_last >= self.snapshot_interval and
                len(self.bid_data) > 0 and
                len(self.ask_data) > 0)

    def validate_orderbook(self):
        """Validate the current order book state"""
        if not self.bid_data or not self.ask_data:
            logger.warning("Order book is empty or incomplete")
            return False

        highest_bid = max(self.bid_data.keys())
        lowest_ask = min(self.ask_data.keys())

        # Check for crossed book
        if highest_bid >= lowest_ask:
            logger.warning(f"Invalid order book: crossed prices (highest bid: {highest_bid}, lowest ask: {lowest_ask})")
            return False

        # Check for reasonable number of levels
        if len(self.bid_data) < 5 or len(self.ask_data) < 5:
            logger.warning(f"Insufficient order book depth (Bid Levels: {len(self.bid_data)}, Ask Levels: {len(self.ask_data)})")
            return False

        return True

    def correct_crossed_prices(self):
        """Correct crossed prices in the order book"""
        if not self.bid_data or not self.ask_data:
            return

        highest_bid = max(self.bid_data.keys())
        lowest_ask = min(self.ask_data.keys())

        if highest_bid >= lowest_ask:
            logger.warning(f"Correcting crossed prices (highest bid: {highest_bid}, lowest ask: {lowest_ask})")
            # Adjust the highest bid and lowest ask to avoid crossing
            self.bid_data.pop(highest_bid, None)
            self.ask_data.pop(lowest_ask, None)

    def reprocess_orderbook(self):
        """Reprocess the order book data with retries if crossed prices are detected"""
        for attempt in range(self.max_retries):
            if self.validate_orderbook():
                return True
            self.correct_crossed_prices()
            time.sleep(self.retry_delay)
        logger.error("Failed to validate order book after retries")
        return False

    def on_message(self, ws, raw_message):
        """Process incoming WebSocket messages"""
        try:
            messages = json.loads(raw_message)
            if isinstance(messages, list) and len(messages) > 1:
                # Process bids
                for b in messages[1].get("b", []):
                    price, volume = float(b[0]), float(b[1])
                    if volume == 0:
                        self.bid_data.pop(price, None)
                    elif price > 0:
                        self.bid_data[price] = volume

                # Process asks
                for a in messages[1].get("a", []):
                    price, volume = float(a[0]), float(a[1])
                    if volume == 0:
                        self.ask_data.pop(price, None)
                    elif price > 0:
                        self.ask_data[price] = volume

                # Reprocess order book if crossed prices are detected
                if self.reprocess_orderbook():
                    # Check if we should take a snapshot (every 10 seconds)
                    if self.should_take_snapshot():
                        mid_price = self.get_mid_price()
                        if mid_price > 0:
                            self.record_current_snapshot(mid_price)
                            self.last_snapshot_time = time.time()
                            self.snapshot_count += 1
                            logger.info(f"Snapshot {self.snapshot_count}/100 recorded at {time.strftime('%H:%M:%S')} | "
                                      f"Price: ${mid_price:,.2f} | "
                                      f"Bid Levels: {len(self.bid_data)} | Ask Levels: {len(self.ask_data)}")
                else:
                    logger.error("Order book validation failed after retries")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def get_mid_price(self):
        """Calculate mid price"""
        if self.bid_data and self.ask_data:
            highest_bid = max(self.bid_data.keys())
            lowest_ask = min(self.ask_data.keys())
            return (highest_bid + lowest_ask) / 2
        return 0

    def get_market_spread(self):
        """Calculate bid-ask spread"""
        try:
            if not self.bid_data or not self.ask_data:
                return 0

            highest_bid = max(self.bid_data.keys())
            lowest_ask = min(self.ask_data.keys())

            if highest_bid > lowest_ask:
                logger.warning("Invalid order book state detected")
                return 0

            spread = lowest_ask - highest_bid
            return max(0, spread)

        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return 0

    def record_current_snapshot(self, mid_price):
        """Store current order book state"""
        bid_copy = self.bid_data.copy()
        ask_copy = self.ask_data.copy()
        self.historical_depth.append((mid_price, bid_copy, ask_copy))

    def prepare_model_input(self):
        """Prepare data for model input"""
        if len(self.historical_depth) < 100:
            return None

        input_data = []
        snapshots = list(self.historical_depth)[-100:]

        for _, bids, asks in snapshots:
            snapshot_data = []

            # Process bids
            bid_prices = sorted(bids.keys(), reverse=True)[:20]
            for price in bid_prices:
                snapshot_data.append([price, bids[price]])
            while len(snapshot_data) < 20:
                snapshot_data.append([0.0, 0.0])

            # Process asks
            ask_prices = sorted(asks.keys())[:20]
            for price in ask_prices:
                snapshot_data.append([price, asks[price]])
            while len(snapshot_data) < 40:
                snapshot_data.append([0.0, 0.0])

            input_data.append(snapshot_data)

        return np.array([input_data])

    def save_snapshots_to_file(self, filename):
        """Save order book history to file"""
        try:
            with open(filename, "w") as f:
                json.dump(list(self.historical_depth), f)
            logger.info(f"Snapshots saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving snapshots: {e}")

class OrderBookAnalyzer:
    def __init__(self, data_manager, model=None):
        logger.info("Initializing analyzer...")
        self.data = data_manager
        self.model = model
        self.min_samples_for_training = 100  # Need at least 100 snapshots

        # Initialize tracking variables
        self.prediction_history = deque(maxlen=100)
        self.last_prediction_time = time.time()
        self.prediction_interval = 10.0  # Match snapshot interval
        self.last_prediction_confidence = None

        # Initialize monitoring
        self.last_gpu_check = time.time()
        self.gpu_check_interval = 5

        self.prediction_accuracy = deque(maxlen=100)  # Track last 100 predictions
        self.last_performance_check = time.time()
        self.performance_check_interval = 60  # Check performance every minute

        # Runtime control
        self.running = True
        logger.info("Analyzer initialization complete")

    def make_prediction(self):
        """Make price movement prediction"""
        try:
            # Check if we have enough data
            if len(self.data.historical_depth) < self.min_samples_for_training:
                logger.info(f"Collecting data: {len(self.data.historical_depth)}/{self.min_samples_for_training} snapshots")
                return None

            model_input = self.data.prepare_model_input()
            if model_input is None:
                logger.info("Model input is None, not enough data for prediction")
                return None

            logger.info(f"Model input shape: {model_input.shape}")

            prediction = self.model.predict(model_input, verbose=0)
            prediction_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])

            # Store prediction and confidence
            self.prediction_history.append(prediction_class)
            self.last_prediction_confidence = float(confidence)

            # Log prediction
            direction_map = {0: "DOWN", 1: "STABLE", 2: "UP"}
            logger.info(
                f"Prediction: {direction_map[prediction_class]} "
                f"(Confidence: {confidence:.2%})"
            )

            # Monitor performance after each prediction
            self.monitor_performance()

            return prediction_class

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def monitor_performance(self):
        """Monitor system performance metrics"""
        try:
            if len(self.prediction_history) < 2:
                return

            # Get the last prediction and actual price movement
            last_prediction = self.prediction_history[-1]

            # Get the last two prices
            recent_prices = [price for price, _, _ in list(self.data.historical_depth)[-2:]]
            actual_movement = 1  # STABLE
            if recent_prices[-1] > recent_prices[-2]:
                actual_movement = 2  # UP
            elif recent_prices[-1] < recent_prices[-2]:
                actual_movement = 0  # DOWN

            # Record if prediction was correct
            correct = (last_prediction == actual_movement)
            self.prediction_accuracy.append(correct)

            # Calculate accuracy
            if self.prediction_accuracy:
                accuracy = sum(self.prediction_accuracy) / len(self.prediction_accuracy)
                logger.info(f"Prediction Accuracy (last {len(self.prediction_accuracy)} predictions): {accuracy:.2%}")

                # Log detailed performance
                predictions_count = len(self.prediction_accuracy)
                correct_count = sum(self.prediction_accuracy)
                logger.info(f"Correct predictions: {correct_count}/{predictions_count}")

                # Log movement distribution
                movement_counts = {
                    "DOWN": sum(1 for p in self.prediction_history if p == 0),
                    "STABLE": sum(1 for p in self.prediction_history if p == 1),
                    "UP": sum(1 for p in self.prediction_history if p == 2)
                }
                logger.info("Prediction Distribution:")
                for direction, count in movement_counts.items():
                    percentage = (count / len(self.prediction_history)) * 100
                    logger.info(f"{direction}: {count} ({percentage:.1f}%)")

        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")

    def log_trading_info(self, mid_price, spread):
        """Log current trading information"""
        try:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            # Show data collection progress
            current_samples = len(self.data.historical_depth)
            if current_samples < self.min_samples_for_training:
                progress = (current_samples / self.min_samples_for_training) * 100
                remaining_samples = self.min_samples_for_training - current_samples
                time_remaining = remaining_samples * 10  # 10 seconds per sample
                minutes = time_remaining // 60
                seconds = time_remaining % 60

                print(f"\nCollecting Data: {current_samples}/{self.min_samples_for_training} ({progress:.1f}%)")
                print(f"Time until predictions: ~{minutes:.0f}m {seconds:.0f}s")
                return

            # Basic price info
            info = [
                f"\nTime: {current_time}",
                f"Price: ${mid_price:,.2f}",
                f"Spread: ${spread:,.2f} ({(spread/mid_price)*100:.3f}%)"
            ]

            # Add market depth info
            if self.data.historical_depth:
                latest = list(self.data.historical_depth)[-1]
                total_bids = sum(latest[1].values())
                total_asks = sum(latest[2].values())

                info.extend([
                    f"Bid Volume: {total_bids:.4f} BTC",
                    f"Ask Volume: {total_asks:.4f} BTC",
                ])

                # Calculate imbalance
                if total_bids + total_asks > 0:
                    imbalance = (total_bids - total_asks) / (total_bids + total_asks)
                    info.append(f"Book Imbalance: {imbalance:+.2%}")

            # Add prediction if available
            if self.prediction_history:
                last_pred = self.prediction_history[-1]
                pred_map = {0: "⬇️", 1: "➡️", 2: "⬆️"}
                conf_str = f" ({self.last_prediction_confidence:.1%})" if self.last_prediction_confidence else ""
                info.append(f"Last Prediction: {pred_map[last_pred]}{conf_str}")

            # Print all info
            print("\n".join(info))
            print("-" * 50)

        except Exception as e:
            logger.error(f"Error logging trading info: {e}")

    def save_trading_data(self, filename="trading_data.json"):
        """Save trading state to file"""
        try:
            # Convert numpy types to native Python types
            def convert_numpy(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, deque):
                    return list(obj)
                return obj

            data = {
                "timestamp": int(time.time()),
                "predictions": [convert_numpy(p) for p in self.prediction_history],
                "prices": [
                    (int(time.time()), float(price))
                    for price, _, _ in self.data.historical_depth
                ],
                "latest_price": float(self.data.get_mid_price()),
                "latest_spread": float(self.data.get_market_spread())
            }

            # Add market stats
            if self.data.historical_depth:
                latest = list(self.data.historical_depth)[-1]
                data["market_stats"] = {
                    "bid_levels": int(len(latest[1])),
                    "ask_levels": int(len(latest[2])),
                    "total_bid_volume": float(sum(latest[1].values())),
                    "total_ask_volume": float(sum(latest[2].values()))
                }

            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Trading data saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving trading data: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def run(self):
        """Main analysis loop"""
        last_save_time = time.time()
        save_interval = 60  # Save every minute
        update_interval = 10  # Match snapshot interval

        try:
            while self.running:
                current_time = time.time()

                # Regular updates
                if current_time - self.last_prediction_time > update_interval:
                    mid_price = self.data.get_mid_price()

                    if mid_price > 0:
                        spread = self.data.get_market_spread()

                        # Log current state
                        self.log_trading_info(mid_price, spread)

                        # Only make predictions if we have enough data
                        if len(self.data.historical_depth) >= self.min_samples_for_training:
                            self.make_prediction()

                            # Only save data once we're making predictions
                            if current_time - last_save_time > save_interval:
                                filename = f"kraken_data/trading_data_{int(current_time)}.json"
                                self.save_trading_data(filename)
                                last_save_time = current_time

                    self.last_prediction_time = current_time

                time.sleep(0.1)  # Prevent CPU overuse

        except KeyboardInterrupt:
            logger.info("Stopping analysis gracefully...")
        except Exception as e:
            logger.error(f"Error in analysis loop: {e}")
        finally:
            if len(self.data.historical_depth) >= self.min_samples_for_training:
                self.save_trading_data("kraken_data/final_trading_data.json")

# WebSocket handlers
def on_open(ws):
    """WebSocket connection opened"""
    logger.info("Connected to Kraken WebSocket")
    subscription = {
        "event": "subscribe",
        "pair": ["BTC/USDT"],
        "subscription": {"name": "book", "depth": 1000}
    }
    ws.send(json.dumps(subscription))
    logger.info("Subscribed to order book")

def on_error(ws, error):
    """WebSocket error handler"""
    logger.error(f"WebSocket error: {error}")

def on_close(ws, status_code, msg):
    """WebSocket connection closed"""
    logger.info(f"WebSocket closed: {status_code} - {msg}")

def build_websocket(data_mgr):
    """Create WebSocket connection"""
    return websocket.WebSocketApp(
        "wss://ws.kraken.com/",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
        on_message=data_mgr.on_message
    )


def main():
    """Main program entry point"""
    try:
        # Load environment variables
        load_dotenv()

        # Create necessary directories
        os.makedirs("kraken_data", exist_ok=True)
        os.makedirs("weights", exist_ok=True)

        logger.info("\nInitializing Neural Network Trading System...")

        # System information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
        logger.info(f"CUDA Available: {tf.test.is_built_with_cuda()}")

        # Initialize model
        logger.info("Creating neural network model...")
        input_shape = (100, 40, 2)  # 100 time steps, 40 features (20 bid + 20 ask levels), 2 values per level
        model = create_deeplob(input_shape)

        if model is None:
            raise ValueError("Failed to create model")

        # Load pre-trained weights if available
        load_model_weights(model)

        # Initialize components
        logger.info("Setting up data manager...")
        data_mgr = DataManager(max_snapshots=200, order_book_depth=500)

        logger.info("Creating market analyzer...")
        analyzer = OrderBookAnalyzer(data_mgr, model=model)

        logger.info("Setting up WebSocket connection...")
        ws = build_websocket(data_mgr)

        # Set up graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nInitiating shutdown sequence...")
            try:
                analyzer.running = False
                ws.close()
                final_snapshot = "kraken_data/final_snapshot.json"
                data_mgr.save_snapshots_to_file(final_snapshot)
                logger.info(f"Final snapshot saved to {final_snapshot}")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("\nStarting trading system...")

        # Start WebSocket in background thread
        ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
        ws_thread.start()

        # Run main analysis loop
        analyzer.run()

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
