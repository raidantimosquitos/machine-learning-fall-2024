# Binary classification model using the Breast Cancer Dataset by M11351802 - Herranz Gancedo, Lucas

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 1) prepare data for training. 
# Create a breast cancer dataset class and scale the data accordingly
class BreastCancerDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        scaler = StandardScaler()
        bc = datasets.load_breast_cancer()
        X_numpy, y_numpy = bc.data, bc.target
        y_numpy = y_numpy.reshape((len(y_numpy),1))
        X_numpy = scaler.fit_transform(X_numpy)
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

bc_dataset = BreastCancerDataset(transform=ToTensor())
bc_train, bc_test = train_test_split(bc_dataset, test_size=0.2, random_state=1234)

# 2) Hyper-parameters and data loaders
input_size = 30
hidden_neuron_num = 15
output_size = 1 # binary classification (cancerigenous or not)
num_epochs = 100
batch_size = 35
learning_rate = 0.001

train_loader = DataLoader(dataset=bc_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=bc_test, batch_size=batch_size, shuffle=False)

# 3) Multi-layer Neural Network model definition, activation functions
# [30 inputs] -> [15 hidden neurons] ->  [1 output]
# one hidden layer
class BinaryClassification(nn.Module):
    def __init__(self, n_input_features, n_hidden_neurons, n_output_neurons):
        super(BinaryClassification, self).__init__()
        self.linear1 = nn.Linear(n_input_features, n_hidden_neurons)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden_neurons, n_output_neurons)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

model = BinaryClassification(n_input_features=input_size, n_hidden_neurons=hidden_neuron_num, n_output_neurons=output_size)
model.to(device=device)

# 4) loss and optimizer definition
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5) training loop (batch training)
history = []
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    loss_per_epoch = []
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass
        output = model(inputs)
        loss = criterion(output, labels)

        loss_per_epoch.append(loss.item())
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 4 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    
    history.append(np.mean(loss_per_epoch))

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        predictions = torch.round(outputs)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct /n_samples
    print("\nTest metrics: ")
    print(f'accuracy = {acc:.4f} %')

plt.plot(history)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Training loss evolution")
plt.show()