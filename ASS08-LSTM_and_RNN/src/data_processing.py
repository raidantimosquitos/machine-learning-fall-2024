import os
import librosa
import numpy as np
import pickle
from config import PROCESSED_DATA_DIR

def extract_mfcc(file_path, n_mfcc=40):
    """Extract MFCC features from an audio file."""
    signal, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Return MFCCs as (time_steps, n_mfcc)

def pad_sequence(features, max_len=100):
    """Pad or truncate the MFCC sequence to a fixed length."""
    if len(features) < max_len:
        padded = np.pad(features, ((0, max_len - len(features)), (0, 0)), mode='constant')
    else:
        padded = features[:max_len]
    return padded

def load_or_compute_mfcc(file_path, n_mfcc=40, max_len=100):
    """Load precomputed MFCCs from processed directory or compute if not found."""
    # Create a unique key for the file
    file_name = os.path.basename(file_path)
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"{file_name}_mfcc.pkl")

    # Check if the MFCCs are already computed and stored
    if os.path.exists(processed_file_path):
        with open(processed_file_path, 'rb') as f:
            mfccs = pickle.load(f)
    else:
        mfccs = extract_mfcc(file_path, n_mfcc)
        mfccs = pad_sequence(mfccs, max_len)
        # Save the computed MFCCs for future use
        with open(processed_file_path, 'wb') as f:
            pickle.dump(mfccs, f)
    
    return mfccs

def process_data(file_path, n_mfcc=40, max_len=100):
    """Process a single file path and return MFCC features."""
    features = load_or_compute_mfcc(file_path, n_mfcc, max_len)
    return features