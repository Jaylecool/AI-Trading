"""
Centralised configuration for the AI Trading System.

Reads values from a .env file (if present) and exposes them as module-level
constants.  Every setting has a sensible default so the system works out of
the box without any .env file at all.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR, '.env'))

# --- Flask ---
FLASK_HOST: str = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT: int = int(os.getenv('FLASK_PORT', '5000'))
FLASK_DEBUG: bool = os.getenv('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')

# --- Trading ---
INITIAL_CAPITAL: float = float(os.getenv('INITIAL_CAPITAL', '100000'))
SUPPORTED_SYMBOLS: list = os.getenv('SUPPORTED_SYMBOLS', 'AAPL,NVDA,MSFT,GOOGL,AMZN,TSLA,META').split(',')

# --- CORS ---
_raw_origins = os.getenv('CORS_ORIGINS', f'http://localhost:{FLASK_PORT},http://127.0.0.1:{FLASK_PORT}')
CORS_ORIGINS: list = [o.strip() for o in _raw_origins.split(',') if o.strip()]

# --- ngrok ---
NGROK_DOMAIN: str = os.getenv('NGROK_DOMAIN', '')

# --- SMTP ---
SMTP_SERVER: str = os.getenv('SMTP_SERVER', '')
SMTP_PORT: int = int(os.getenv('SMTP_PORT', '587'))
SMTP_SENDER: str = os.getenv('SMTP_SENDER', '')
SMTP_PASSWORD: str = os.getenv('SMTP_PASSWORD', '')

# --- Streaming ---
STREAM_UPDATE_FREQUENCY: int = int(os.getenv('STREAM_UPDATE_FREQUENCY', '2'))

# --- Auth ---
import secrets as _secrets
SECRET_KEY: str = os.getenv('SECRET_KEY', _secrets.token_hex(32))

# --- Directories ---
RESULTS_DIR: str = os.path.join(_BASE_DIR, 'results')
TRAINED_MODELS_DIR: str = os.path.join(_BASE_DIR, 'trained_models')
DATA_DIR: str = os.path.join(_BASE_DIR, 'data')
DATABASE_PATH: str = os.path.join(DATA_DIR, 'users.db')
