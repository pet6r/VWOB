# Vaporwave Order Book

A vaporwave-themed 3D wave visualization for orderbook depth analysis

https://github.com/user-attachments/assets/cf93d6a6-c2ef-4407-9328-4635fd454427

<p align="center"><em>running kraken_mountain.py</em></p>

## Supported Exchanges

- **Kraken** — No authentication required
- **Coinbase** — No authentication required
- **Binance** — No authentication required (not available in US)
- **Alpaca** — Requires API key and secret in `.env` file (see `.env_example`)

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and sync

```bash
git clone https://github.com/pet6r/VWOB.git
cd VWOB
uv sync
```

This creates the `.venv` and installs all dependencies from `pyproject.toml`.

## Usage

```bash
# 3D mountain orderbook (default: XBT/USD)
uv run kraken_mountain.py

# Specify a different pair
uv run kraken_mountain.py --pair ETH/USD

# Original vaporwave visualizer
uv run kraken_visuals.py
```

### Other exchanges

```bash
uv run coinbase_visuals.py
uv run binance_visuals.py
uv run alpaca_visuals.py   # requires .env
```

## Controls (kraken_mountain)

| Key | Action |
|-----|--------|
| Arrow Keys | Rotate camera |
| W / A / S / D | Fly through scene |
| + / - | Zoom in / out |
| [ / ] | Adjust orderbook depth levels |
| I | Toggle HUD stats |
| R | Reset camera |
