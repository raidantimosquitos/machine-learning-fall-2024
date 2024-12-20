# Initialize the src package

# Optionally, you can expose commonly used functions or modules directly from src
from .data_processing import extract_mfcc, pad_sequence, process_data
from .model import LSTM_RNN_Model
from .generate_labels import extract_labels
from .utils import save_model_checkpoint