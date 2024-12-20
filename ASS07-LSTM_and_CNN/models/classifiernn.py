'''
classifiernn.py

'''

# import local libraries
from .convnn import CNNModule
from .lstm import LSTMModule

# import third-party libraries
import torch
import torch.nn as nn

# Define the combined model with CNN, LSTM, and a classifier
class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        self.cnn = CNNModule()
        self.lstm = LSTMModule(28, 128, 1)
        self.fc = nn.Linear(128 + 128, 10)  # Concatenate CNN and LSTM outputs

    def forward(self, x):
        cnn_out = self.cnn(x)
        lstm_out = self.lstm(x)
        combined = torch.cat((cnn_out, lstm_out), dim=1)
        return self.fc(combined)