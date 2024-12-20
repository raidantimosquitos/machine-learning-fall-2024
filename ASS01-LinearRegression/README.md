Machine Learning ASSIGNMENT01 Deliverable: Linear regression model
==================================================================================

This code belongs to M11351802 - Herranz Gancedo, Lucas.

The *linear_regression.py* code has two main functions:
1. Creates and trains a basic regression model using PyTorch tensors.
2. Output predictions and resiudual plot using matplotlib Python library.

## Introduction
------------------------------
Linear regression is a simple yet powerful statistical method used for modeling the relationship between a dependent variable (target) and one or more independent variables (features). In its simplest form, univariate linear regression (one-dimensional input feature), the goal is to find a straight line that best fits the data, minimizing the error between predicted and actual values.

The model can be represented as:
$$\displaystyle y = mx + c$$
where:
- $y$: predicted value (dependent variable).
- $x$: input feature (independent variable).
- $m$: slope of the line (coefficient).
- $c$: intercept.

Linear regression assumes a linear relationship between $x$ and $y$ and is commonly solved using methods like least squares optimization.

For further, reading you can see:
- [Linear regression explained - Towards Data Science](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86)
- [An Introduction to statistical learning](https://www.statlearning.com/) (Chapter 3)

## Code structure
----------------------------------------------------------
**Step 0:** Training Dataset generation, using numpy to create random samples to construct training dataset

**Step 1:** Design the model:
- Input size.
- Output size.
- Training forward pass (no. of layers ~1 and operations y = w*x + b).

**Step 2:** Define the loss:
- We use Mean Square Error as loss function, thus loss should follow MSE = (1/N) * (w*x + b - y)**2

**Step 3:** Training loop:
- Forward pass (compute prediction)
- Backward pass (gradient calculation)
- Update model weigths

**Bonus** Plots

## Required libraries to run the code
----------------------------------------

I only used three libraries: [PyTorch](https://pytorch.org/get-started/locally/), [Numpy](https://numpy.org/doc/stable/reference/) and [Matplotlib.pyplot](https://matplotlib.org/stable/api/pyplot_summary.html) package.

```Python
import torch
import numpy as np
import matplotlib.pyplot as plt
```

## Step 0: Training data generation
------------------------------------------------

Create X_numpy as a numpy array of 100 samples of value between -1 and 1. Shuffle it to randomize the data.
Create Y_numpy as a function of X and following a normal distribution.

```Python
# 0) Data generation
X_numpy = np.linspace(-1, 1, 100)[:,np.newaxis]
np.random.shuffle(X_numpy)    # randomize the data
Y_numpy =  3 * X_numpy + 2 + np.random.normal(loc=0, scale=0.5, size=(100, 1))
```

Create Pytorch tensors from the previous data and shape them according to our needs.

```Python
# Create tensors from numpy type datsets.
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

# Reshape y tensor from [100] to [100, 1] in order to perform operations 
Y = Y.view(Y.shape[0],1)
```

Create Pytorch tensors for the model ($y=w*x+b$) parameters, set the requires_grad field to True, since we will compute the gradient of the loss with respect to these two parameters. Also get feature and sample number from the input data.

```Python
# Create model parameter tensors, initialize on 0.0, set requires_grad to True
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

n_samples, n_features = X.shape
```

## Step 1: Design the model (input size, output size, forward pass - all operations / layers)
-----------------------------------------------------------------------------------------------
Define the operations / transformations that the model should perform (y = w*x + b, in this case). Get the tensor input and output sizes.

```Python
input_size = n_features
output_size = 1

# forward pass          
def forward(w, x, b):
    return w*x + b
```

## Step 2: Define the loss function
--------------------------------------
Define the loss function. Mean square error in our case: loss = MSE = (1/N) * (w*x + b - y)**2

```Python
# 2) Define the loss (and optimizer if required)
# loss = MSE = (1/N) * (w*x + b - y)**2
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()
```

## Step 3: Training loop
---------------------------
Set the model training rules (learning rate and no. of epochs). The pipeline is as follows:
- compute the prediction of the model from input data (forward pass).
- compute the loss of prediction with respect to training output.
- compute the gradients of previous loss with respect to 'w' and 'b'.
- update the parameters ('w' and 'b').
- set the gradients to zero.
- print samples of training steps

```Python
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
```

## Bonus: Plot
------------------------
Plotting the trained model equation and the residual plot.

![Loss plots](img/Figure1.png)

```Python
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
```