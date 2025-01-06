import websocket
import json

def on_message(ws, message):
    """Callback for when a message is received from the server."""
    data = json.loads(message)
    print("Order Book Update:")
    print("Bids:", data['b'])
    print("Asks:", data['a'])

def on_error(ws, error):
    """Callback for when an error occurs."""
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    """Callback for when the WebSocket connection is closed."""
    print("WebSocket Closed")

def on_open(ws):
    """Callback for when the WebSocket connection is opened."""
    print("WebSocket Connected")
    # You can optionally perform actions after the connection is established.

if __name__ == "__main__":
    # Replace 'btcusdt' with the symbol you want to subscribe to
    symbol = "btcusdt"
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@depth"

    # Create a WebSocketApp instance
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    # Set the callback for on_open
    ws.on_open = on_open

    # Run the WebSocket in a loop
    ws.run_forever()
