# Constructing a Pytorch multi-class classifier for the iris dataset
# 1) Design our model (input size, output size, forward pass - all operations / layers)
# 
# 2) Construct the loss and optimizer
#
# 3) Training loop
#   -- forward pass: compute prediction
#   -- backward pass: gradients
#   -- update weights
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Custom hinge loss
def hinge_loss(outputs, targets):
    margin = 1 - targets * outputs
    loss = torch.mean(torch.clamp(margin, min=0))
    return loss

# 0) prepare the data
# Load the Iris dataset
iris_dataset = datasets.load_iris()
X, y = iris_dataset.data, iris_dataset.target

# Convert the problem to binary classification: Setosa (0) vs the rest (1)
y_binary = (y != 0).astype(int)

# Split into train and test sets
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# create pytorch tensors
X_train = torch.tensor(X_train, dtype= torch.float32)
y_train = torch.tensor(y_train, dtype= torch.float32).view(-1,1)
X_test = torch.tensor(X_test, dtype= torch.float32)
y_test = torch.tensor(y_test, dtype= torch.float32).view(-1,1)

# 1) model
# [4 inputs] -> [8 hidden neurons] -> [3 outputs]
# one hidden layer only
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# define the SVM model
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

model = SVM(input_dim=n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.HingeEmbeddingLoss() # Hinge Embedding loss 
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)

# 3) training loop
num_epochs = 500

for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = hinge_loss(y_predicted, y_train)
    # zero gradients
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}')


with torch.no_grad():
    outputs = model(X_test)
    predictions = (outputs > 0).float()
    accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test.numpy(), predictions.numpy(), target_names=["Setosa", "Other"], labels=[0,1]))