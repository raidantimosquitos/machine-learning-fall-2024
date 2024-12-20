import torch
import numpy as np
import matplotlib.pyplot as plt

# 0) Data generation
X_numpy = np.linspace(-1, 1, 100)[:,np.newaxis]
np.random.shuffle(X_numpy)    # randomize the data
Y_numpy =  3 * X_numpy + 2 + np.random.normal(loc=0, scale=0.5, size=(100, 1))

# Create tensors from numpy type datsets.
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

# Reshape y tensor from [100] to [100, 1] in order to perform operations 
Y = Y.view(Y.shape[0],1)

# Create model parameter tensors, initialize on 0.0, set requires_grad to True
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

n_samples, n_features = X.shape

# 1) Design our model (input size, output size, forward pass - all operations / layers)
input_size = n_features
output_size = 1

# forward pass          
def forward(w, x, b):
    return w*x + b

# 2) Define the loss (and optimizer if required)
# loss = MSE = (1/N) * (w*x + b - y)**2
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# 3) Training loop
#   -- forward pass: compute prediction
#   -- backward pass: gradients
#   -- update weights

learning_rate = 0.01
num_epochs = 1000

for epoch in range(num_epochs):
    # prediction of the model from input data
    y_pred = forward(X, w, b)

    # compute loss with respect to training output
    l = loss(Y, y_pred)

    # gradients = backward pass. Gradient of the loss 'l' with respect to the weights
    l.backward()

    # update our weights if not using gradients
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # set  gradients to zero, so new ones can be computed on next iteration
    w.grad.zero_()
    b.grad.zero_()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, b = {b:.3f}, loss = {l:.3f}')

# plot
figure, axis = plt.subplots(2, 1)
predicted = y_pred.detach().numpy()
residuals = Y_numpy - predicted
axis[0].plot(X_numpy, Y_numpy, 'ro')
axis[0].plot(X_numpy, predicted, 'b')
axis[0].set_title('Training samples and output model')
axis[0].set_xlabel('Actual values')
axis[0].set_ylabel('Predicted values')
axis[0].grid()
axis[1].scatter(predicted,residuals,alpha=0.6)
axis[1].set_title('Residual plot')
axis[1].set_xlabel('Predicted values')
axis[1].set_ylabel('Residuals')
axis[1].grid()
plt.show()
