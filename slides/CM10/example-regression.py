import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from tabicl import TabICLRegressor

np.random.seed(42)

n_samples = 100
x_train = np.random.rand(n_samples) * 6 - 3
e = np.random.normal(scale=0.3, size=n_samples)
y_train = np.sin(4*x_train) + np.cos(2*x_train) + e

x_test = np.linspace(-3, 3, 1000)
f = np.sin(4*x_test) + np.cos(2*x_test)

y_pred_dict = {}

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train.reshape(-1, 1), y_train)
y_pred_knn = knn.predict(x_test.reshape(-1, 1))
y_pred_dict["KNN (5)"] = y_pred_knn

hgb = HistGradientBoostingRegressor(random_state=0)
hgb.fit(x_train.reshape(-1, 1), y_train)
y_pred_hgb = hgb.predict(x_test.reshape(-1, 1))
y_pred_dict["HistGradientBoosting"] = y_pred_hgb

tabicl = TabICLRegressor()
tabicl.fit(x_train.reshape(-1, 1), y_train)
y_pred_tabicl = tabicl.predict(x_test.reshape(-1, 1))
y_pred_dict["TabICL"] = y_pred_tabicl

fig, ax = plt.subplots(figsize=(13.25, 4.4), ncols=3, sharey=True)
plt.subplots_adjust(wspace=0.05)
for i, (model, y_pred) in enumerate(y_pred_dict.items()):
    ax[i].plot(x_test, f, label="true", lw=2.0, color='k')
    ax[i].scatter(x_train, y_train, c='C0', label="data", s=10)
    ax[i].plot(x_test, y_pred, label="predicted", lw=2.0, color='C1')
    ax[i].set_title(model)
ax[-1].legend()
fig.show()
