import os

# Directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CURATED_DATA_DIR = os.path.join(DATA_DIR, "curated")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(CURATED_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Data collection settings
CAPTURE_WIDTH = 1366
CAPTURE_HEIGHT = 768
CAPTURE_TOP = 100
CAPTURE_LEFT = 0
UDP_PORT = 5300

# Preprocessing settings
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
CROP_TOP_PERCENT = 33
FRAME_STACK_SIZE = 5

# Training settings
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001 