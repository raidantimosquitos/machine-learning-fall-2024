'''
convnn.py

'''
# import third-party libraries
import torch
import torch.nn as nn

class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the output size after convolutions and pooling.
        # The input size is 28x28, and after two Conv2d+MaxPool2d layers,
        # the output size will be reduced. We compute this to set the fc layer.        
        self._calculate_conv_output_size()

        self.fc = nn.Linear(self.conv_out_size, 128)

    def _calculate_conv_output_size(self):
        # Start with the input size of 28x28
        input_size = 28
        # Apply conv1 (kernel_size=3, padding=1), followed by max pool (kernel_size=2, stride=2)
        output_size = (input_size - 3 + 2 * 1) // 2 + 1  # Conv1 + Pooling
        # Apply conv2 (kernel_size=3, padding=1), followed by max pool (kernel_size=2, stride=2)
        output_size = (output_size - 3 + 2 * 1) // 2 + 1  # Conv2 + Pooling
        # Final output size after Conv2 and Pooling will be output_size x output_size x 64 channels
        self.conv_out_size = 64 * output_size * output_size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x