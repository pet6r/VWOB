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
import psutil
import websocket_handler  # imports the C++ websocket handler


# Suppress TensorFlow warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR)

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

# Set up logging
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
            show_level=False
        )
    ]
)
logger = logging.getLogger("rich")

# Import other dependencies
from sklearn.model_selection import train_test_split
import websocket
from dotenv import load_dotenv

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
    weights_file = f"{weights_path}.weights.h5"
    if os.path.exists(weights_file):
        logger.info("Loading saved weights...")
        model.load_weights(weights_file)
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
                # merge both incremental updates and snapshot updates
                b_updates = messages[1].get("b", []) + messages[1].get("bs", [])
                a_updates = messages[1].get("a", []) + messages[1].get("as", [])

                # process bids
                for b in b_updates:
                    price, volume = float(b[0]), float(b[1])
                    if volume == 0:
                        self.bid_data.pop(price, None)
                    elif price > 0:
                        self.bid_data[price] = volume

                # process asks
                for a in a_updates:
                    price, volume = float(a[0]), float(a[1])
                    if volume == 0:
                        self.ask_data.pop(price, None)
                    elif price > 0:
                        self.ask_data[price] = volume

                # reprocess order book if crossed prices are detected
                if self.reprocess_orderbook():
                    # check if we should take a snapshot (every 10 seconds)
                    if self.should_take_snapshot():
                        mid_price = self.get_mid_price()
                        if mid_price > 0:
                            self.record_current_snapshot(mid_price)
                            self.last_snapshot_time = time.time()
                            self.snapshot_count += 1
                            logger.info(
                                f"Snapshot {self.snapshot_count}/100 recorded at "
                                f"{time.strftime('%H:%M:%S')} | Price: ${mid_price:,.2f} | "
                                f"Bid Levels: {len(self.bid_data)} | Ask Levels: {len(self.ask_data)}"
                            )
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

        # Add checkpoint settings
        self.checkpoint_dir = "weights"
        self.best_accuracy = 0.0
        self.checkpoint_interval = 100  # Save every 100 predictions
        self.prediction_count = 0

        # Add PnL tracking
        self.initial_price = None
        self.total_pnl = 0.0
        self.trades = []

        # ... existing initialization ...
        self.trades_won = 0
        self.total_trades = 0
        self.winning_streak = 0
        self.longest_winning_streak = 0
        self.current_streak = 0

        # Add a timer for periodic analysis
        self.last_analysis_time = time.time()
        self.analysis_interval = 30  # Run analysis every 30 seconds

        # Runtime control
        self.running = True
        logger.info("Analyzer initialization complete")

    def track_pnl(self, current_price):
        """Track profit/loss from predictions"""
        if self.initial_price is None:
            self.initial_price = current_price
            return

        if len(self.prediction_history) > 0:
            last_pred = self.prediction_history[-1]
            price_change = current_price - self.initial_price

            if last_pred == 2 and price_change > 0:  # Correct UP prediction
                self.total_pnl += abs(price_change)
            elif last_pred == 0 and price_change < 0:  # Correct DOWN prediction
                self.total_pnl += abs(price_change)

            self.trades.append({
                'timestamp': time.time(),
                'price': current_price,
                'prediction': ['DOWN', 'STABLE', 'UP'][last_pred],
                'price_change': price_change,
                'running_pnl': self.total_pnl
            })

    def analyze_market_trend(self):
        """Analyze market trend over different timeframes"""
        try:
            if len(self.data.historical_depth) < 50:
                return

            prices = [price for price, _, _ in self.data.historical_depth]

            # Short-term trend (last 10 snapshots)
            short_term = np.mean(prices[-10:]) - np.mean(prices[-20:-10])

            # Medium-term trend (last 50 snapshots)
            medium_term = np.mean(prices[-25:]) - np.mean(prices[-50:-25])

            logger.info("\nMarket Trend Analysis:")
            logger.info(f"Short-term trend: {'UP' if short_term > 0 else 'DOWN'} ({abs(short_term):.2f})")
            logger.info(f"Medium-term trend: {'UP' if medium_term > 0 else 'DOWN'} ({abs(medium_term):.2f})")

        except Exception as e:
            logger.error(f"Error analyzing market trend: {e}")

    def save_model_weights(self, checkpoint_name="latest"):
        """Save model weights"""
        try:
            # Ensure checkpoint directory exists
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Add the required .weights.h5 extension
            filepath = os.path.join(self.checkpoint_dir, f"model_checkpoint_{checkpoint_name}.weights.h5")
            self.model.save_weights(filepath)
            logger.info(f"✅ Model weights saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving model weights: {e}")
            return False

    def make_prediction(self):
        """Make price movement prediction"""
        try:
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

            # Get detailed prediction probabilities
            direction_map = {0: "DOWN", 1: "STABLE", 2: "UP"}
            probabilities = {direction_map[i]: float(prediction[0][i]) for i in range(3)}

            # Log detailed prediction information
            logger.info("\n" + "-" * 50)
            logger.info("Prediction Details:")
            logger.info(f"Direction: {direction_map[prediction_class]} ({confidence:.1%})")
            logger.info("Probabilities:")
            for direction, prob in probabilities.items():
                logger.info(f"  {direction}: {prob:.1%}")

            # Get price movement since last prediction
            if len(self.data.historical_depth) >= 2:
                last_two_prices = [price for price, _, _ in list(self.data.historical_depth)[-2:]]
                price_change = (last_two_prices[1] - last_two_prices[0])
                price_change_pct = (price_change / last_two_prices[0]) * 100
                logger.info(f"Price Change: ${price_change:.2f} ({price_change_pct:+.3f}%)")

            logger.info("-" * 50)

            return prediction_class

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def analyze_order_book_depth(self):
        """Analyze the order book depth and liquidity"""
        if not self.bid_data or not self.ask_data:
            return

        # Calculate total volume at each side
        total_bid_volume = sum(self.bid_data.values())
        total_ask_volume = sum(self.ask_data.values())

        # Calculate imbalance
        volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

        # Calculate spread
        best_bid = max(self.bid_data.keys())
        best_ask = min(self.ask_data.keys())
        spread = best_ask - best_bid
        spread_percentage = (spread / best_bid) * 100

        logger.info(f"\nOrder Book Analysis:")
        logger.info(f"Bid Levels: {len(self.bid_data)}")
        logger.info(f"Ask Levels: {len(self.ask_data)}")
        logger.info(f"Total Bid Volume: {total_bid_volume:.4f} BTC")
        logger.info(f"Total Ask Volume: {total_ask_volume:.4f} BTC")
        logger.info(f"Volume Imbalance: {volume_imbalance:+.2%}")
        logger.info(f"Spread: ${spread:.2f} ({spread_percentage:.3f}%)")

        # Analyze depth distribution
        bid_volume_distribution = self.analyze_depth_distribution(self.bid_data, best_bid, ascending=False)
        ask_volume_distribution = self.analyze_depth_distribution(self.ask_data, best_ask, ascending=True)

        logger.info("\nDepth Distribution:")
        logger.info("Bids:")
        for range_str, volume in bid_volume_distribution.items():
            logger.info(f"  {range_str}: {volume:.4f} BTC")
        logger.info("Asks:")
        for range_str, volume in ask_volume_distribution.items():
            logger.info(f"  {range_str}: {volume:.4f} BTC")

    def analyze_depth_distribution(self, data, reference_price, ascending=True, ranges=[0.1, 0.5, 1.0, 2.0, 5.0]):
        """Analyze volume distribution at different price ranges"""
        distribution = {}

        for percentage in ranges:
            if ascending:
                price_limit = reference_price * (1 + percentage/100)
                range_str = f"Up to +{percentage}%"
            else:
                price_limit = reference_price * (1 - percentage/100)
                range_str = f"Down to -{percentage}%"

            volume = sum(vol for price, vol in data.items()
                        if (price <= price_limit if ascending else price >= price_limit))
            distribution[range_str] = volume

        return distribution

    def monitor_performance(self):
        """Monitor system performance metrics"""
        try:
            if len(self.prediction_history) < 2:
                return

            # Get the last prediction and actual price movement
            last_prediction = self.prediction_history[-1]

            # Calculate actual movement from recent prices
            recent_prices = [price for price, _, _ in list(self.data.historical_depth)[-2:]]
            actual_movement = 1  # STABLE
            if recent_prices[-1] > recent_prices[-2]:
                actual_movement = 2  # UP
            elif recent_prices[-1] < recent_prices[-2]:
                actual_movement = 0  # DOWN

            # Record if prediction was correct
            correct = (last_prediction == actual_movement)
            self.prediction_accuracy.append(correct)

            # Update win rate statistics
            self.total_trades += 1
            if correct:
                self.trades_won += 1
                self.current_streak += 1
                if self.current_streak > self.longest_winning_streak:
                    self.longest_winning_streak = self.current_streak
            else:
                self.current_streak = 0

            # Calculate and log performance metrics
            if self.prediction_accuracy:
                accuracy = sum(self.prediction_accuracy) / len(self.prediction_accuracy)
                win_rate = (self.trades_won / self.total_trades) if self.total_trades > 0 else 0

                # Create detailed performance log
                logger.info("\nPerformance Update:")
                logger.info(f"Overall Accuracy: {accuracy:.2%}")
                logger.info(f"Win Rate: {win_rate:.2%}")
                logger.info(f"Trades Won: {self.trades_won}/{self.total_trades}")
                logger.info(f"Current Streak: {self.current_streak}")
                logger.info(f"Longest Winning Streak: {self.longest_winning_streak}")
                logger.info(f"Recent Prediction: {'Correct ✅' if correct else 'Incorrect ❌'}")

                # Calculate rolling accuracy (last 10 predictions)
                recent_accuracy = sum(list(self.prediction_accuracy)[-10:]) / min(10, len(self.prediction_accuracy))
                logger.info(f"Recent Accuracy (last 10): {recent_accuracy:.2%}")

                # Direction distribution
                movement_counts = {
                    "DOWN": sum(1 for p in self.prediction_history if p == 0),
                    "STABLE": sum(1 for p in self.prediction_history if p == 1),
                    "UP": sum(1 for p in self.prediction_history if p == 2)
                }

                logger.info("\nPrediction Distribution:")
                for direction, count in movement_counts.items():
                    percentage = (count / len(self.prediction_history)) * 100
                    logger.info(f"{direction}: {count} ({percentage:.1f}%)")

                # Log actual vs predicted
                logger.info(f"\nLast Prediction vs Actual:")
                logger.info(f"Predicted: {['DOWN', 'STABLE', 'UP'][last_prediction]}")
                logger.info(f"Actual: {['DOWN', 'STABLE', 'UP'][actual_movement]}")

                price_change = recent_prices[-1] - recent_prices[-2]
                price_change_pct = (price_change / recent_prices[-2]) * 100
                logger.info(f"Price Change: ${price_change:.2f} ({price_change_pct:+.3f}%)")

                logger.info("-" * 50)

        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")

    def log_market_conditions(self):
        """Log detailed market conditions"""
        try:
            if not self.data.historical_depth:
                return

            latest_data = list(self.data.historical_depth)[-1]
            mid_price = latest_data[0]
            bids = latest_data[1]
            asks = latest_data[2]

            logger.info("\nMarket Conditions:")
            logger.info(f"Current Price: ${mid_price:,.2f}")

            # Volume analysis
            total_bid_volume = sum(bids.values())
            total_ask_volume = sum(asks.values())
            volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

            logger.info(f"Bid Volume: {total_bid_volume:.4f} BTC")
            logger.info(f"Ask Volume: {total_ask_volume:.4f} BTC")
            logger.info(f"Volume Imbalance: {volume_imbalance:+.2%}")

            # Order book depth
            logger.info(f"Bid Levels: {len(bids)}")
            logger.info(f"Ask Levels: {len(asks)}")

            # Price levels analysis
            top_bids = sorted(bids.items(), reverse=True)[:5]
            top_asks = sorted(asks.items())[:5]

            logger.info("\nTop 5 Bid Levels:")
            for price, volume in top_bids:
                logger.info(f"${price:,.2f}: {volume:.4f} BTC")

            logger.info("\nTop 5 Ask Levels:")
            for price, volume in top_asks:
                logger.info(f"${price:,.2f}: {volume:.4f} BTC")

            logger.info("-" * 50)

        except Exception as e:
            logger.error(f"Error logging market conditions: {e}")


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

                console.print(f"\n[bold yellow]Collecting Data: {current_samples}/{self.min_samples_for_training} ({progress:.1f}%)[/bold yellow]")
                console.print(f"[bold yellow]Time until predictions: ~{minutes:.0f}m {seconds:.0f}s[/bold yellow]")
                return

            # Basic price info
            info = [
                f"\n[bold cyan]Time: {current_time}[/bold cyan]",
                f"[bold cyan]Price: ${mid_price:,.2f}[/bold cyan]",
                f"[bold cyan]Spread: ${spread:,.2f} ({(spread/mid_price)*100:.3f}%)[/bold cyan]"
            ]

            # Add market depth info
            if self.data.historical_depth:
                latest = list(self.data.historical_depth)[-1]
                total_bids = sum(latest[1].values())
                total_asks = sum(latest[2].values())

                info.extend([
                    f"[bold cyan]Bid Volume: {total_bids:.4f} BTC[/bold cyan]",
                    f"[bold cyan]Ask Volume: {total_asks:.4f} BTC[/bold cyan]",
                ])

                # Calculate imbalance
                if total_bids + total_asks > 0:
                    imbalance = (total_bids - total_asks) / (total_bids + total_asks)
                    info.append(f"[bold cyan]Book Imbalance: {imbalance:+.2%}[/bold cyan]")

            # Add prediction if available
            if self.prediction_history:
                last_pred = self.prediction_history[-1]
                pred_map = {0: "⬇️", 1: "➡️", 2: "⬆️"}
                conf_str = f" ({self.last_prediction_confidence:.1%})" if self.last_prediction_confidence else ""
                info.append(f"[bold cyan]Last Prediction: {pred_map[last_pred]}{conf_str}[/bold cyan]")

            # Print all info
            console.print("\n".join(info))
            console.print("[bold cyan]" + "-" * 50 + "[/bold cyan]")

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
        last_weight_save_time = time.time()
        last_performance_check = time.time()

        save_interval = 60  # Save trading data every minute
        weight_save_interval = 300  # Save weights every 5 minutes
        update_interval = 10  # Match snapshot interval
        performance_check_interval = 60  # Check performance every minute

        prediction_count = 0
        best_accuracy = 0.0

        try:
            while self.running:
                current_time = time.time()

                # Regular updates
                if current_time - self.last_prediction_time > update_interval:
                    mid_price = self.data.get_mid_price()

                    if mid_price > 0:

                        # Run periodic analysis
                        if current_time - self.last_analysis_time > self.analysis_interval:
                            self.analyze_market_condition()
                            self.last_analysis_time = current_time
                        spread = self.data.get_market_spread()
                        current_samples = len(self.data.historical_depth)

                        # Log current state
                        self.log_trading_info(mid_price, spread)

                        # Check if we have enough data for predictions and training
                        if current_samples >= self.min_samples_for_training:
                            prediction = self.make_prediction()
                            prediction_count += 1

                            # Weight saving logic - only execute after minimum samples
                            if current_time - last_weight_save_time > weight_save_interval:
                                logger.info(f"Saving weights after {prediction_count} predictions...")
                                self.save_model_weights(f"periodic_{prediction_count}")
                                last_weight_save_time = current_time

                            # Performance monitoring - only after minimum samples
                            if current_time - last_performance_check > performance_check_interval:
                                if self.prediction_accuracy:
                                    current_accuracy = sum(self.prediction_accuracy) / len(self.prediction_accuracy)
                                    logger.info(f"Current accuracy: {current_accuracy:.2%}")

                                    if current_accuracy > best_accuracy:
                                        best_accuracy = current_accuracy
                                        self.save_model_weights("best")
                                        logger.info(f"New best accuracy: {current_accuracy:.2%} - Saved weights")

                                last_performance_check = current_time

                            # Save trading data - only after minimum samples
                            if current_time - last_save_time > save_interval:
                                filename = f"kraken_data/trading_data_{int(current_time)}.json"
                                self.save_trading_data(filename)
                                last_save_time = current_time
                                self.log_performance_metrics()
                        else:
                            # Log data collection progress
                            remaining_samples = self.min_samples_for_training - current_samples
                            logger.info(f"Collecting data: {current_samples}/{self.min_samples_for_training} "
                                        f"({remaining_samples} more needed before predictions start)")

                        self.last_prediction_time = current_time

                time.sleep(0.1)  # Prevent CPU overuse

        except KeyboardInterrupt:
            logger.info("Stopping analysis gracefully...")
            if len(self.data.historical_depth) >= self.min_samples_for_training:
                self.save_model_weights("interrupt")
        except Exception as e:
            logger.error(f"Error in analysis loop: {e}")
            if len(self.data.historical_depth) >= self.min_samples_for_training:
                self.save_model_weights("error")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Only save final state if we collected enough data
            if len(self.data.historical_depth) >= self.min_samples_for_training:
                prediction = self.make_prediction()
                self.monitor_performance()
                self.log_market_conditions()
            else:
                logger.warning(f"Program terminated before collecting minimum samples "
                                f"({len(self.data.historical_depth)}/{self.min_samples_for_training})")

    def analyze_market_condition(self):
            """Comprehensive market analysis"""
            try:
                logger.info("\n" + "="*50)
                logger.info("Market Analysis Report")
                logger.info("="*50)

                # Analyze order book depth
                self.analyze_order_book_depth()

                # Analyze market trend
                self.analyze_market_trend()

                # Log market conditions
                self.log_market_conditions()

                # Add timestamp
                logger.info(f"\nAnalysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("="*50 + "\n")

            except Exception as e:
                logger.error(f"Error in market condition analysis: {e}")

    def analyze_order_book_depth(self):
        """Analyze the order book depth and liquidity"""
        try:
            if not self.data.bid_data or not self.data.ask_data:  # Fixed reference
                logger.warning("Order book is empty")
                return

            # Calculate total volume at each side
            total_bid_volume = sum(self.data.bid_data.values())  # Fixed reference
            total_ask_volume = sum(self.data.ask_data.values())  # Fixed reference

            # Calculate imbalance
            total_volume = total_bid_volume + total_ask_volume
            if total_volume > 0:
                volume_imbalance = (total_bid_volume - total_ask_volume) / total_volume
            else:
                volume_imbalance = 0

            # Calculate spread
            best_bid = max(self.data.bid_data.keys())
            best_ask = min(self.data.ask_data.keys())
            spread = best_ask - best_bid
            spread_percentage = (spread / best_bid) * 100 if best_bid > 0 else 0

            logger.info(f"\nOrder Book Analysis:")
            logger.info(f"Bid Levels: {len(self.data.bid_data)}")
            logger.info(f"Ask Levels: {len(self.data.ask_data)}")
            logger.info(f"Total Bid Volume: {total_bid_volume:.4f} BTC")
            logger.info(f"Total Ask Volume: {total_ask_volume:.4f} BTC")
            logger.info(f"Volume Imbalance: {volume_imbalance:+.2%}")
            logger.info(f"Spread: ${spread:.2f} ({spread_percentage:.3f}%)")

            # Analyze depth distribution
            bid_volume_distribution = self.analyze_depth_distribution(self.data.bid_data, best_bid, ascending=False)
            ask_volume_distribution = self.analyze_depth_distribution(self.data.ask_data, best_ask, ascending=True)

            logger.info("\nDepth Distribution:")
            logger.info("Bids:")
            for range_str, volume in bid_volume_distribution.items():
                logger.info(f"  {range_str}: {volume:.4f} BTC")
            logger.info("Asks:")
            for range_str, volume in ask_volume_distribution.items():
                logger.info(f"  {range_str}: {volume:.4f} BTC")

        except Exception as e:
            logger.error(f"Error analyzing order book depth: {e}")

    def analyze_depth_distribution(self, data, reference_price, ascending=True, ranges=[0.1, 0.5, 1.0, 2.0, 5.0]):
        """Analyze volume distribution at different price ranges"""
        distribution = {}
        try:
            for percentage in ranges:
                if ascending:
                    price_limit = reference_price * (1 + percentage/100)
                    range_str = f"Up to +{percentage}%"
                else:
                    price_limit = reference_price * (1 - percentage/100)
                    range_str = f"Down to -{percentage}%"

                volume = sum(vol for price, vol in data.items()
                            if (price <= price_limit if ascending else price >= price_limit))
                distribution[range_str] = volume

        except Exception as e:
            logger.error(f"Error in depth distribution analysis: {e}")

        return distribution
    def log_performance_metrics(self):
        """Log detailed performance metrics"""
        try:
            if self.prediction_accuracy and self.prediction_history:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                # Calculate overall accuracy
                accuracy = sum(self.prediction_accuracy) / len(self.prediction_accuracy)
                win_rate = (self.trades_won / self.total_trades) if self.total_trades > 0 else 0

                # Calculate direction distribution
                total_predictions = len(self.prediction_history)
                down_count = sum(1 for p in self.prediction_history if p == 0)
                stable_count = sum(1 for p in self.prediction_history if p == 1)
                up_count = sum(1 for p in self.prediction_history if p == 2)

                # Create performance report
                report = [
                    f"\n[bold cyan]Performance Metrics ({current_time}):[/bold cyan]",
                    f"Accuracy: {accuracy:.2%}",
                    f"Win Rate: {win_rate:.2%}",
                    f"Total Trades: {self.total_trades}",
                    f"Trades Won: {self.trades_won}",
                    f"Current Streak: {self.current_streak}",
                    f"Longest Winning Streak: {self.longest_winning_streak}",
                    f"Total Predictions: {total_predictions}",
                    "\nDirection Distribution:",
                    f"  DOWN: {down_count} ({down_count/total_predictions:.1%})",
                    f"  STABLE: {stable_count} ({stable_count/total_predictions:.1%})",
                    f"  UP: {up_count} ({up_count/total_predictions:.1%})"
                ]

                if self.last_prediction_confidence:
                    report.append(f"Last Prediction Confidence: {self.last_prediction_confidence:.2%}")

                # Add PnL information if available
                if hasattr(self, 'total_pnl'):
                    report.append(f"Total PnL: ${self.total_pnl:,.2f}")

                console.print("\n".join(report))
                console.print("[bold cyan]" + "-" * 50 + "[/bold cyan]")

        except Exception as e:
            logger.error(f"Error logging performance metrics: {e}")

    def monitor_system_resources(self):
        """Monitor system resource usage"""
        try:

            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            if cpu_percent > 80 or memory.percent > 80:
                logger.warning(f"High resource usage - CPU: {cpu_percent}%, Memory: {memory.percent}%")

        except ImportError:
            pass  # psutil not available
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")


    def get_market_summary(self):
        """Get a brief summary of current market conditions"""
        try:
            mid_price = self.data.get_mid_price()
            spread = self.data.get_market_spread()

            return {
                'price': mid_price,
                'spread': spread,
                'bid_levels': len(self.data.bid_data),
                'ask_levels': len(self.data.ask_data),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return None

# WebSocket handlers
def on_open(ws):
    """WebSocket connection opened"""
    logger.info("Connected to Kraken WebSocket")
    subscription = {
        "event": "subscribe",
        "pair": ["XBT/USDT"],
        "subscription": {"name": "book", "depth": 1000}
    }
    ws.send_message(json.dumps(subscription))
    logger.info("Subscribed to order book")

def on_error(ws, error):
    """WebSocket error handler"""
    logger.error(f"WebSocket error: {error}")

def on_close(ws, status_code, msg):
    """WebSocket connection closed"""
    logger.info(f"WebSocket closed: {status_code} - {msg}")

def build_websocket(data_mgr):
    ws = websocket_handler.WebSocketHandler()

    def on_message(msg):
        logger.debug(f"Received message: {msg[:200]}...")
        data_mgr.on_message(None, msg)

    ws.set_message_callback(on_message)

    sub_cmd = {
        "event": "subscribe",
        "pair": ["XBT/USDT"],
        "subscription": {"name": "book", "depth": 1000}
    }

    ws.set_initial_subscription(json.dumps(sub_cmd))
    ws.connect("wss://ws.kraken.com/")

    # Wait for connection and initial subscription to complete
    retry_count = 0
    max_retries = 5
    while retry_count < max_retries:
        time.sleep(2)  # Wait between checks
        if ws.is_connected():
            logger.info("WebSocket connection established successfully")

            # Wait a bit more for the connection to stabilize
            time.sleep(2)

            # Now check latency
            #latency = ws.get_initial_latency()
            #logger.info(f"Latency: {latency:.2f}ms")

            #if latency > 1000:  # More than 1 second
            #    logger.warning("High latency detected in WebSocket connection!")

            return ws

        retry_count += 1
        logger.info(f"Waiting for connection... (attempt {retry_count}/{max_retries})")

    logger.error("Failed to establish WebSocket connection after multiple attempts")
    return None




def main():
    """Main program entry point"""
    try:
        # Load environment variables
        load_dotenv()

        # Create necessary directories
        os.makedirs("kraken_data", exist_ok=True)
        os.makedirs("weights", exist_ok=True)

        console.print("\n[bold green]Initializing Neural Network Trading System...[/bold green]")

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

        # Set up WebSocket connection
        logger.info("Setting up WebSocket connection...")
        ws = build_websocket(data_mgr)  # create websocket handler
        #ws.connect("wss://ws.kraken.com/")  # connect once

        # Add connection check
        time.sleep(2)  # Wait for connection to establish
        if not ws.is_connected():
            logger.error("Failed to establish WebSocket connection")
            return

        logger.info("WebSocket connection established successfully")

        # Define periodic task (e.g., log system status)
        def periodic_task():
            while analyzer.running:  # keep running while the analyzer is active
                time.sleep(60)  # sleep for 60 seconds
                logger.info("Periodic check: System is running")

        # Run periodic task in a background thread
        periodic_thread = threading.Thread(target=periodic_task, daemon=True)
        periodic_thread.start()

        # Set up graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nInitiating shutdown sequence...")
            try:
                analyzer.running = False  # stop the analyzer
                ws.stop()  # stop the websocket
                logger.info("WebSocket stopped successfully.")

                # Save final model weights and snapshots
                analyzer.save_model_weights("emergency")
                final_snapshot = "kraken_data/final_snapshot.json"
                data_mgr.save_snapshots_to_file(final_snapshot)
                logger.info(f"Final snapshot saved to {final_snapshot}")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            finally:
                sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        console.print("\n[bold green]Starting trading system...[/bold green]")

        # Run the main analysis loop
        analyzer.run()

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
