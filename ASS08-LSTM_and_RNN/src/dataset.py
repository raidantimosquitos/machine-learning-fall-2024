import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_processing import process_data

class RAVDESSDataset(Dataset):
    """
    PyTorch Dataset for the RAVDESS audio dataset.
    Reads file paths and labels from a labels.csv file.
    """
    def __init__(self, labels_csv, raw_data_dir, n_mfcc=40, max_len=100):
        """
        Args:
        - labels_csv (str): Path to the labels.csv file.
        - raw_data_dir (str): Directory containing the raw audio files.
        - n_mfcc (int): Number of MFCCs to extract.
        - max_len (int): Maximum length to pad/truncate the MFCC features.
        """
        self.raw_data_dir = raw_data_dir
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        
        # Load the labels.csv file
        if not os.path.exists(labels_csv):
            raise ValueError(f"Labels file {labels_csv} not found.")
        
        self.data = pd.read_csv(labels_csv)
        self.labels_map = {label: idx for idx, label in enumerate(self.data['emotion'].unique())}
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve an audio sample and its label by index.
        Returns:
        - features (torch.Tensor): MFCC features (max_len x n_mfcc).
        - label (torch.Tensor): One-hot encoded label.
        """
        # Get the file name and emotion label
        row = self.data.iloc[idx]
        file_name = row['filename']
        label = row['emotion']
        
        # Full path to the audio file
        file_path = os.path.join(self.raw_data_dir, file_name)
        
        # Process the audio file to extract MFCC features
        mfcc_features = process_data(file_path, n_mfcc=self.n_mfcc, max_len=self.max_len)
        
        # Convert label to an index and one-hot encode it
        label_idx = self.labels_map[label]
        one_hot_label = np.zeros(len(self.labels_map))
        one_hot_label[label_idx] = 1
        
        return torch.tensor(mfcc_features, dtype=torch.float32), torch.tensor(one_hot_label, dtype=torch.float32)

    def get_all_labels(self, as_one_hot=False):
        """
        Retrieve all labels in the dataset.
        
        Args:
        - as_one_hot (bool): Whether to return labels as one-hot encoded arrays. Defaults to False.
        
        Returns:
        - List of labels as class indices or one-hot encoded arrays.
        """
        if as_one_hot:
            return [torch.tensor(self.__getitem__(idx)[1], dtype=torch.float32) for idx in range(len(self))]
        else:
            return [torch.argmax(self.__getitem__(idx)[1]).item() for idx in range(len(self))]