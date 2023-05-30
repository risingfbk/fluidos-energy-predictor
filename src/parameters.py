# Dataset parameters
# Span of a single step in minutes
from src.support.dt import WEEK_IN_MINUTES

GRANULARITY = 15
# Offset in minutes between each file
OFFSET = 24 * 60

STEPS_IN = WEEK_IN_MINUTES // GRANULARITY
STEPS_OUT = 1
N_FEATURES = 2
SPLIT = 0.2

FILTERS = 144
KSIZE = 3

OVERHEAD = 1.05

DROPOUT = 0
PATIENCE = 100

LEARNING_RATE = 0.01

LOG_FOLDER = "out"

DEFAULT_MODEL = "model1"
MODEL_FOLDER = "models"
GCD_FOLDER = "data/gcd"
CACHE_FOLDER = "data/cache"

TEST_FILE_AMOUNT = 24
TRAIN_FILE_AMOUNT = 24
