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
# Defaults match the saved checkpoints (128/2).  To use a smaller VPS model
# set these in .env AND delete model/checkpoints/*.pt before restarting:
#   HIDDEN_SIZE=64  NUM_LAYERS=1  SEQUENCE_LENGTH=20
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "30"))     # sliding window size
HIDDEN_SIZE     = int(os.getenv("HIDDEN_SIZE",     "128"))    # LSTM hidden units
NUM_LAYERS      = int(os.getenv("NUM_LAYERS",      "2"))      # LSTM layers
DROPOUT = 0.3
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))               # larger batch = fewer gradient steps
EPOCHS_INITIAL = 100                                           # for initial bulk training
ONLINE_LR = 0.0005                                            # lower LR for online updates

# --- VPS / CPU throttling ---
# Number of PyTorch intra-op threads.  1-2 is best on a shared-core VPS
# (avoids thread contention and memory spikes during training).
TORCH_THREADS    = int(os.getenv("TORCH_THREADS",    "1"))
# Epochs used at server startup (checkpoint is already well-trained; just warm-up).
STARTUP_EPOCHS   = int(os.getenv("STARTUP_EPOCHS",   "3"))
# Epochs and interval for the background continuous trainer.
CONTINUOUS_EPOCHS    = int(os.getenv("CONTINUOUS_EPOCHS",    "5"))
CONTINUOUS_INTERVAL  = int(os.getenv("CONTINUOUS_INTERVAL",  "3600"))  # seconds (default 60 min)

# --- Prediction ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

# --- Feature Engineering ---
ROLLING_WINDOWS = [5, 10, 20, 50]
MARKOV_ORDER = 4                                               # Markov chain memory depth

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.getenv("STORAGE_DIR", BASE_DIR)  # Docker: /app/storage, Local: project root
DB_PATH = os.path.join(STORAGE_DIR, "data", "predictions.db")
MODEL_DIR = os.path.join(STORAGE_DIR, "model", "checkpoints")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- API Poller (replaces Selenium scraper) ---
API_BASE_URL = "https://draw.ar-lottery01.com"
API_POLL_INTERVALS = {
    "30sec": 8,   # seconds between polls
    "1min": 15,
    "3min": 30,
}
API_BACKFILL_ON_START = True  # fetch initial batch of results on startup

# --- Legacy Scraper (deprecated, kept for reference) ---
SCRAPER_POLL_INTERVAL = 5
SITE_URL = "https://91clu.org/"
