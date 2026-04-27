import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from tabicl import TabICLClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Generate simulated data for classification
X, y = make_moons(n_samples=1000, noise=0.35, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Plot the data to get a sense of it
fig, ax = plt.subplots(figsize=(7.4, 7.2))
scatter0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], c='C0', label="Class 0")
scatter1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], c='C1', label="Class 1")
ax.legend(["Class 0", "Class 1"])
ax.set_title("Data for Classification")
fig.show()

models_dict = {}
accuracy_dict = {}

# Create a K-Nearest Neighbors classifier and fit it to the training data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Evaluate the K-Nearest Neighbors model on the test set
accuracy_knn = knn.score(X_test, y_test)
accuracy_dict["KNN (5)"] = accuracy_knn
models_dict["KNN (5)"] = knn

# Create HistGradientBoostingClassifier and fit it to the training data
hgb = HistGradientBoostingClassifier(random_state=0)
hgb.fit(X_train, y_train)
# Evaluate the HistGradientBoostingClassifier model on the test set
accuracy_hgb = hgb.score(X_test, y_test)
accuracy_dict["HistGradientBoosting"] = accuracy_hgb
models_dict["HistGradientBoosting"] = hgb

# Create a TabICLClassifier and fit it to the training data
tabicl = TabICLClassifier()
tabicl.fit(X_train, y_train)
# Evaluate the model on the test set
accuracy_tabicl = tabicl.score(X_test, y_test)
accuracy_dict["TabICL"] = accuracy_tabicl
models_dict["TabICL"] = tabicl

# Print the accuracy of each model
for model_name, accuracy in accuracy_dict.items():
    print(f"{model_name} Accuracy: {accuracy:.4f}")

fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True, ncols=3)

for i, (model_name, model) in enumerate(models_dict.items()):

    # Create a mesh to plot decision boundaries
    h = 0.2
    offset = 0.5
    x_min, x_max = X[:, 0].min() - offset, X[:, 0].max() + offset
    y_min, y_max = X[:, 1].min() - offset, X[:, 1].max() + offset
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict probabilities on mesh
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    ax[i].contourf(xx, yy, Z, levels=20, cmap="RdYlBu_r", alpha=0.8)
    ax[i].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    # Plot training data
    scatter = ax[i].scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="RdYlBu_r",
        edgecolors="k",
        s=50,
        alpha=0.8,
    )

    ax[i].set(xlabel="Feature 1", ylabel="Feature 2")
    ax[i].set_title(f"{model_name}")

fig.suptitle("Decision Boundaries and Test Data", fontsize=16)
fig.colorbar(scatter, ax=ax, label="Probability of class 1")
fig.show()
