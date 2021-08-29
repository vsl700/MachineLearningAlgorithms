import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from math import sqrt

import abalones_dataset_for_knn

X, y = abalones_dataset_for_knn.get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)  # random_state - seed

knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
print('root-mean squared error:', rmse)

test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
print('root-mean squared error:', rmse)

# Checking model fitting (the color of the points is the predicted value, while the axes are length and diameter)
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=test_preds, s=50, cmap=cmap  # Changing the arguments will let you know more
)
f.colorbar(points)
plt.show()
