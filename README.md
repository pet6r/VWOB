# Vaporwave Order Book

A vaporwave-themed 3D wave visualization for orderbook depth analysis *(Currently in progress)*
<p align="center">
  <img src="https://github.com/user-attachments/assets/ecd9f621-72e8-4b4b-98ce-06935424a6bc" alt="vwob gif" width="550">
</p>
<p align="center"><em>running kraken_visuals.py</em></p>


## 🔧 Supported Exchanges

- ✅ **Kraken US** - No authentication required
- ✅ **Coinbase US** - No authentication required
- ✅ **Binance** - No authentication required (Not available in US)
- ✅ **Alpaca** - Requires API key and secret in `.env` file

## 📦 Installation

### Basic Dependencies
```bash
# Python packages
pip install PyQt5 vispy websocket-client numpy python-dotenv

# System dependencies (Ubuntu/Debian)
sudo apt install python3-pyqt5
```

### CNN Model Dependencies
```bash
# Python packages
pip install tensorflow keras scikit-learn tensorflow[and-cuda] pybind11 numpy

# System dependencies (Ubuntu/Debian)
sudo apt install pybind11-dev libboost-all-dev
```

## 🚀 Additional Setup

### CUDA Installation
Visit the official NVIDIA CUDA downloads page:
- [CUDA Downloads for Linux x86_64 Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)
- [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip)

### WebSocket Connection
This project uses [@zaphoyd's websocketpp](https://github.com/zaphoyd/websocketpp) for the websocket connection (CNN model)

---
