'''
lstm.py

'''

# import local libraries
from models.hyperparameters import DEVICE

# import third-party libraries
import torch
import torch.nn as nn

class LSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)  # Reshape for LSTM input
        h0 = torch.zeros(1, x.size(0), 128).to(DEVICE)
        c0 = torch.zeros(1, x.size(0), 128).to(DEVICE)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        return torch.relu(x)