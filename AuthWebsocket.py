import websocket
import json
import time
import base64
import hashlib
import hmac
import os

# API Connection to environment variables
API_KEY = os.getenv('apikey')
API_SECRET = os.getenv('secretkey')

# Exchange nouce requirment function
def get_signature(api_path, nonce, post_data, secret):
    message = (str(nonce) + post_data).encode()
    secret = base64.b64decode(secret)
    hashed = hmac.new(secret, message, hashlib.sha512)
    return base64.b64encode(hashed.digest()).decode()

def on_message(ws, message):
    print("Message received:", message)

def on_open(ws):
    # Generate nonce
    nonce = int(time.time() * 1000)

    # Define subscription message for private data
    subscribe_data = {
        "event": "subscribe",
        "subscription": {"name": "ownTrades"},
        "nonce": nonce
    }

    # Serialize and create the signature
    post_data = json.dumps(subscribe_data)
    signature = get_signature('/private/ownTrades', nonce, post_data, API_SECRET)

    # Add authentication details
    subscribe_data.update({
        "api_key": API_KEY,
        "api_signature": signature
    })

    # Send subscription message
    ws.send(json.dumps(subscribe_data))

# Connect to Kraken WebSocket API
url = "wss://ws-auth.kraken.com/"  # Private WebSocket endpoint
ws = websocket.WebSocketApp(url, on_message=on_message, on_open=on_open)
ws.run_forever()
