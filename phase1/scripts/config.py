"""Shared paths and experimental constants. Portable across machines."""
import os

# Project root = parent of scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 42
EPOCHS = 5          # baseline pilot budget
FT_EPOCHS = 3       # same FT budget for all pruning configs
BATCH_SIZE = 64
LR = 0.1
FT_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0

# Thread control: override with env NUM_THREADS if you want
NUM_THREADS = int(os.environ.get("NUM_THREADS", "2"))

# Latency protocol
WARMUP = 20
MEASURE_ITERS_BS1 = 100
MEASURE_ITERS_BS32 = 50
