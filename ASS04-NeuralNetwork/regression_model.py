# Regression model using the Diabetes dataset by M11351802 - Herranz Gancedo, Lucas

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import copy
import matplotlib.pyplot as plt

# 0) GPU device selection (if avaliable)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
# device = "cpu"
print(f"Using {device} device")

# 1) prepare data for training. 
# Create a diabetes dataset class and scale the data accordingly
class DiabetesDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        diabetes = datasets.load_diabetes()
        sc = MinMaxScaler()
        X_numpy, y_numpy = diabetes.data, diabetes.target
        y_numpy = y_numpy.reshape(-1,1)
        y_numpy = sc.fit_transform(y_numpy)
        self.X = X_numpy.astype(np.float32)
        self.y = y_numpy.astype(np.float32)
        self.n_samples = X_numpy.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        # dataset[index]
        sample = self.X[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples


class ToTensor():
    def __call__(self, sample):
        features, targets = sample
        return torch.from_numpy(features), torch.from_numpy(targets)

diabetes_dataset = DiabetesDataset(transform=ToTensor())
diabetes_train, diabetes_test = train_test_split(diabetes_dataset, test_size=0.2, random_state=1234)

# 2) Hyper-parameters and dataloaders
input_size = 10
hidden_neuron_num = 4
output_size = 1
num_epochs = 100
batch_size = 10
learning_rate = 1e-3

train_loader = DataLoader(dataset=diabetes_train, batch_size=batch_size, shuffle=True)

# 3) Multi-layer Neural Network model definition, activation functions
# [10 inputs] -> [4 hidden neurons] -> [1 output]
# a single hidden layer
class LinearRegression(nn.Module):
    def __init__(self, n_input_features, n_output_neurons, n_hidden_neurons):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(n_input_features, n_hidden_neurons)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden_neurons, n_output_neurons)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = LinearRegression(n_input_features=input_size, n_output_neurons=output_size, n_hidden_neurons=hidden_neuron_num)
model.to(device=device)

# 4) loss and optimizer definition
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Hold the best model
best_mse = np.inf   #init to infinity
best_weights = None
history = []

# 5) training loop (batch training)
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward pass
        output = model(inputs)
        loss = criterion(output, targets)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (i + 1) % 6 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

    # evaluate accuracy at end of each epoch
    test_in, test_targets = zip(*[diabetes_test[j] for j in range(len(diabetes_test))])
    test_in = torch.stack(test_in).to(device)
    test_targets = torch.stack(test_targets).to(device)
    y_pred = model(test_in)
    mse = criterion(y_pred, test_targets)
    mse = float(mse)
    history.append(mse)

    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# Print the best evaluation metrics
print("\nEvaluation metrics: ")    
print(f"\tMSE: {best_mse:.4f}")
print(f"\tRMSE: {np.sqrt(best_mse):.4f}")
print()

model.load_state_dict(best_weights)

# plot evaluation metrics
plt.plot(history)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Evaluation loss evolution")
plt.show()