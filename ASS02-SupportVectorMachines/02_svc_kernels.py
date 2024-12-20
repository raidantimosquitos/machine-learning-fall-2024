import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# 0) prepare the iris dataset
bc = load_iris()
X, y = bc.data, bc.target

y = y.reshape((len(y),1))

# For this demonstration, I will just run SVM on the Petal length and width (the last two features), 
# and build a setosa vs the rest classifier. Constructing the training data:
X = [[x[2], x[3]] for x in X]

X = np.asanyarray(X, dtype=np.float32)

for i in range(len(y)):
    if y[i] == 0: y[i] = 1
    else: y[i] = -1

y = np.asarray(y, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Plotting settings
fig_1, axes_1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharey=True)
x_min, x_max, y_min, y_max = min(X_train[:,0]) - 1, max(X_train[:,0]) + 1, min(X_train[:,1]) - 1, max(X_train[:,1]) + 1
axes_1.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

# Plot samples by color and add legend
scatter = axes_1.scatter(X_train[:, 0], X_train[:, 1], s=150, c=y_train, label=y_train, edgecolors="k")
axes_1.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
axes_1.set_title("Samples in two-dimensional feature space")
axes_1.set_xlabel("Petal length [cm]")
axes_1.set_ylabel("Petal width [cm]")
#_ = plt.show()

# define the models with different kernels
svc_rbf = SVC(kernel="rbf", gamma=2)
svc_lin = SVC(kernel="linear", gamma=2)
svc_poly = SVC(kernel="poly", gamma=2)
svc_sigm = SVC(kernel="sigmoid", gamma=2)

svcs = [svc_lin, svc_poly, svc_rbf, svc_sigm]
kernel_label = ["Linear", "Polynomial", "RBF", "Sigmoid"]
model_color = ["m", "c", "g", "r"]
lw = 2

fig_2, axes_2 = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharey=True)
fig_2.suptitle("SVM classification model for Iris dataset (versicolour [1] vs the rest [-1]) using only Petal length and width variables", fontsize='x-large', fontweight='bold')
# plotting training data with decision boundaries
def plot_training_data_with_decision_boundary(
    model, ax=axes_2, long_title=True, support_vectors=True
):
    # Train the SVC
    clf = model.fit(X_train,y_train.ravel())
    print(f"model's accuracy with {model.kernel} kernel: {accuracy_score(y_test.ravel(), clf.predict(X_test)):.3f}")
    
    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharey=True)
    x_min, x_max, y_min, y_max = min(X_train[:,0]) - 1, max(X_train[:,0]) + 1, min(X_train[:,1]) - 1, max(X_train[:,1]) + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X_train, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_xlabel("Petal length [cm]")
    ax.set_ylabel("Petal width [cm]")
    if long_title:
        ax.set_title(f" Decision boundaries of {model.kernel} kernel in SVC")
    else:
        ax.set_title(model.kernel)

    
for ix, svc in enumerate(svcs):
    if ix < 2:
        ax_idx = [int(i) for i in list("0" + "{0:b}".format(ix))]
    else:
        ax_idx = [int(i) for i in list("{0:b}".format(ix))]
    plot_training_data_with_decision_boundary(svc, ax=axes_2[ax_idx[0],ax_idx[1]])
# end of for loop

plt.show()