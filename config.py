"""
config.py - every setting for train_good_job.py lives here. Edit, then run:

    python train_good_job.py
"""

# ---------------- data / output ----------------
SPLITS_DIR = "splits"          # from split_and_sample.py (train.json, val.json, test.json)
OUT_DIR = "runs/run"           # best.pt, tokenizer/, config.json, history.json, ...

# ---------------- labels ----------------
TYPE_LABELS = ["ML JOB", "JOB", "OTHER", "NOT JOB"]
BOARD_LABELS = ["Regular account", "Job board"]
BOARD_VALUE = "Job board"      # `poster` value meaning job board; anything else = regular

# ---------------- model ----------------
MODEL = "bert-base-uncased"
MAX_LEN = 128
USE_AUTHOR = False             # True: feed `author [SEP] post_text`

# ---------------- optimisation ----------------
EPOCHS = 5
PATIENCE = 10                   # early stop after this many epochs without val improvement
LR = 3e-5                      # encoder
HEAD_LR = 1e-3                 # classification heads
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
WARMUP = 0.1                   # fraction of steps

# ---------------- multi-task loss ----------------
LAMBDA_TYPE = 1.0
LAMBDA_BOARD = 1.0

# ---------------- class imbalance ----------------
IMBALANCE = "weights"           # job-type head, pick ONE: "none" | "weights" | "sampler"
CLASS_POWER = 1.0              # exponent for IMBALANCE = "weights"
BOARD_WEIGHTS = False          # inverse-frequency weights for the job-board head

# ---------------- model selection / evaluation ----------------
SELECT_ON = "type"             # which val macro-F1 picks best.pt: "type" | "board" | "both"
EVAL_TEST = False              # evaluate on test.json (do this ONCE, at the end)
SEED = 42