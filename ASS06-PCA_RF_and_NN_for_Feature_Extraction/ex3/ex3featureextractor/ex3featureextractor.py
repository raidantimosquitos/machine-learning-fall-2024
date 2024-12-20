'''
ex3featureextractor.py
This Python script, defines the featureExtractor model and performs the
training of the model.
'''
# In-built libraries
import copy

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 20
NUM_EPOCHS = 1000

# Feature Extractor deep neural network definition.   
class featureExtractor(nn.Module):
    def __init__(self, n_input_features, output_size):
        super(featureExtractor, self).__init__()
        self.hidden1 = nn.Linear(n_input_features, 16*n_input_features)
        self.hidden2 = nn.Linear(16*n_input_features, 8*n_input_features)
        self.act = nn.ReLU()
        self.output = nn.Linear(8*n_input_features, output_size)

    def forward(self, x):
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.act(self.output(x))
        return x
    
# This function extracts the feature of a given dataset to later be fed to the SVM classifier
def feature_extraction(dataset):
    n_samples = len(dataset)
    n_features = 4

    print(f'Starting dataset shape: {n_samples}, {n_features}')
    
    # 1) Create a feature extraction model object
    output_size = 16
    extractor_model = featureExtractor(n_input_features=n_features, output_size=output_size)
    extractor_model.float()
    extractor_model.to(DEVICE)

    # 2) hyperparameters, loss and optimizer
    learning_rate = 0.001

    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss() # cross-entropy loss 
    optimizer = torch.optim.Adam(extractor_model.parameters(), lr=learning_rate)

    # 3) training loop
    n_total_steps = len(dataloader)
    history = []
    best_cel = np.inf   #init to infinity
    best_weights = None

    print('Starting the training for the feature extractor model...')
    for epoch in range(NUM_EPOCHS):
        loss_per_epoch = []
        for i, (inputs, labels) in enumerate(dataloader):
            inputs.float()
            labels.float()
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # forward pass
            output = extractor_model(inputs)
            loss = criterion(output, labels)
            loss_per_epoch.append(loss.item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((epoch + 1) % 50 == 0) and ((i + 1) % 2 == 0):
                print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
        
        # record loss value per each training epoch
        history.append(np.mean(loss_per_epoch))
        
        # keep record of the lowest loss and related model weights
        if history[len(history) - 1] < best_cel:
            best_cel = history[len(history) - 1]
            best_weights = copy.deepcopy(extractor_model.state_dict())

    plt.plot(history)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Feature Extractor model - training loss evolution")

    # 4) Print the best evaluation metrics
    print("\nFeature extractor model, best evaluation metrics: ")    
    print(f"\tCross-entropy loss: {best_cel:.4f}")
    print()

    extractor_model.load_state_dict(best_weights)
    # ensure dataset_features is sent to DEVICE if necessary
    dataset_features = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(DEVICE)
    # use extractor_model to generate extracted features
    extracted_features = extractor_model(dataset_features).detach().cpu()
    # construct features and labels tensors
    features, labels = extracted_features, torch.stack([dataset[i][1] for i in range(len(dataset))])

    # 5) Return the features and labels of the output of the best performing feature extractor
    return features, labels