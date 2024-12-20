import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Typical Pytorch Model
# 1) Design our model (input size, output size, forward pass - all operations / layers)
# 
# 2) Construct the loss and optimizer
#
# 3) Training loop
#   -- forward pass: compute prediction
#   -- backward pass: gradients
#   -- update weights

# 0) prepare the data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale the features, remove the mean and scale to the standard deviation of the array
def standardScaler(input_data):
    for col in range(input_data.shape[1]):
        input_data[:,col] = (input_data[:,col] - np.mean(input_data[:,col]))/np.std(input_data[:,1])

    return input_data

X_train = standardScaler(X_train)
X_test = standardScaler(X_test)

y_train = y_train.T
y_test = y_test.T


class LogisticRegression:
    # define Logistic Regression class constructor and intialize variables first
    def __init__(self, learning_rate=0.001, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = np.zeros([n_samples,1], dtype=np.float32)
        self.bias = 0
    
    # define the training of the model
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate output variable (y) with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) #derivative w.r.t weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  #derivative w.r.t bias
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

itr=[10, 100, 500, 1000, 2000]
acc=[]
regressor = []
predictions = []
counter = 0

for i in itr:
    regressor.append(LogisticRegression(learning_rate=0.001, n_iters=i))
    regressor[counter].fit(X_train, y_train)
    predictions.append(regressor[counter].predict(X_train))
    print(f"Logistic Regressor Model trained for {i} epochs, training accuracy: {accuracy(y_train, predictions[counter]):.4f}")
    acc.append(accuracy(y_train, predictions[counter]))
    print(f"Test dataset accuracy: {accuracy(y_test, regressor[counter].predict(X_test)):.4f}")
    print()
    counter = counter + 1


plt.scatter(itr,acc,color="r")
plt.plot(itr,acc)
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.show()