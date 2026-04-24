"""
Central configuration for the Win Go prediction system.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- 91Club Credentials ---
PHONE_NUMBER = os.getenv("PHONE_NUMBER", "")
PASSWORD = os.getenv("PASSWORD", "")

# --- Game Settings ---
GAME_INTERVAL = int(os.getenv("GAME_INTERVAL", "3"))  # minutes per round

# --- Server ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# --- Model Hyperparameters ---
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "30"))     # sliding window size
HIDDEN_SIZE = 128                                              # LSTM hidden units
NUM_LAYERS = 2                                                 # LSTM layers
DROPOUT = 0.3
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
BATCH_SIZE = 32
EPOCHS_INITIAL = 100                                           # for initial bulk training
ONLINE_LR = 0.0005                                            # lower LR for online updates

# --- Prediction ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

# --- Feature Engineering ---
ROLLING_WINDOWS = [5, 10, 20, 50]
MARKOV_ORDER = 4                                               # Markov chain memory depth

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "predictions.db")
MODEL_DIR = os.path.join(BASE_DIR, "model", "checkpoints")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Scraper ---
SCRAPER_POLL_INTERVAL = 5  # seconds between page polls
SITE_URL = "https://91clu.org/"
