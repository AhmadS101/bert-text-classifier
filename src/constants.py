import torch

""" Configuration File """

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512
DROPOUT_RATE = 0.3
FREEZE_LAYERS = True
FREEZE_EMBEDDINGS = True
FREEZE_EARLY_LAYERS = 6

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

VALIDATION_SPLIT = 0.2
DATASET_NAME = "fancyzhx/ag_news"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


RANDOM_SEED = 42
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
