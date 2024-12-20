'''
ex3datasets.py
This script contains all dataset related classes and functions. A first Iris dataset
class which loads the dataset from sklearn.datasets library, and returns it as a list
of tuples of input features and output target. Two transformations are defined, one
to standarize the features of the dataset and another to return features and targets as
tensors.
'''

# Third-party libraries
import numpy as np
from sklearn.datasets import load_iris
import torch
from torch.utils.data import Dataset

# Iris dataset class, loaded from sklearn.datasets module.
# 3 target classes (setosa, versicolor and virginica), 150 samples in total
# IrisDataset class with transformations to standardize features and output
# a tensor of them.

class IrisDataset(Dataset):
    def __init__(self, transform=None, feature_scaling=None):
        iris = load_iris()
        self.X = iris.data.astype(np.float32)  # Features
        self.y = iris.target
        self.n_samples = self.X.shape[0]
        self.transform = transform
        self.feature_scaling = feature_scaling
        # Apply scaling to the features if scaler is provided
        if self.feature_scaling:
            self.X = np.array([self.feature_scaling(features) for features in self.X])
    
    def __getitem__(self, index):
        sample = self.X[index], self.y[index]
        
        # Apply transformations if available
        if self.transform:
            sample = self.transform(sample)

        return sample  # Returning as (features, one-hot encoded target)

    def __len__(self):
        return self.n_samples

# StandardScaler transformation class
class StandardScaler:
    def __init__(self, dataset):
        # Access the data in the dataset and calculate mean and std for scaling
        features = dataset.data  # Extract features directly from the Bunch object
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)

    def __call__(self, features):
        # Apply standard scaling (z-score normalization)
        scaled_features = (features - self.mean) / self.std
        return scaled_features.astype(np.float32)

# ToTensor transformation class. Returns input samples as torch tensors.
class ToTensor:
    def __call__(self, sample):
        features, target = sample
        # Convert features to a tensor (from numpy array)
        features = torch.from_numpy(features)
        # Convert target to a tensor if it's not an array
        target = torch.tensor(target) if not isinstance(target, np.ndarray) else torch.from_numpy(target)
        
        return features, target
    
# Define a custom dataset to wrap the extracted features and original labels.
# Here we set the binary classification problem (Setosa vs the rest)
class ExtractedFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        labels[labels==2] = 1
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.labels[idx]
        sample = features, target
        return sample