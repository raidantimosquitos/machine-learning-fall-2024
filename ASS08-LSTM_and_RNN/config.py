import os

# Define global constants
CWD = os.getcwd()
RAW_DATA_DIR = CWD + "/data/raw/"
PROCESSED_DATA_DIR = CWD + "/data/processed/"
LABELS_FILE = CWD + "/data/labels.csv"
CHECKPOINTS_DIR = CWD + "/checkpoints/"
LOGS_DIR = CWD + "/logs/"