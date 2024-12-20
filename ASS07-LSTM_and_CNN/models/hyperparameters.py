import torch

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'