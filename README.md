# Visual Wave Order Book

Vaporwave Themed 3D Wave Visualization for orderbook depth analysis, Currently in progress

Kraken US and Coinbase US visualizations work without needing an auth/apikey

Binance also works without an auth/apikey but will not work if you are in the US

Alpaca requires an apikey and secret to be set in an .env file

![image](https://github.com/user-attachments/assets/32edec6b-3f94-4cc0-b2a0-b7daa8679c2d)

# Dependencies

```bash
pip install PyQt5 vispy websocket-client numpy python-dotenv
```

# For CNN model fun

```bash
pip install tensorflow keras sklearn tensorflow[and-cuda]
```

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local
https://www.tensorflow.org/install/pip
