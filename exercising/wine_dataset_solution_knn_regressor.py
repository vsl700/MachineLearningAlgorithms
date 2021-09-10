from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics
from math import sqrt


def print_scores(y_true, y_pr):
    print('RMSE:', sqrt(metrics.mean_squared_error(y_true, y_pr)))


data = datasets.load_wine()
print(len(data.data[:, 0]))  # 178

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.35, random_state=123)

clf = KNeighborsRegressor(n_neighbors=3)  # train error is 0 with distance weights
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print_scores(y_test, y_pred)  # RMSE: 0.5968760532369259

y_pred = clf.predict(X_train)
print_scores(y_train, y_pred)  # RMSE: 0.3768673314407159
print('\n')

# Improving kNN
parameters = {"n_neighbors": range(1, 50),
              "weights": ["uniform", "distance"]}
grid_search = GridSearchCV(KNeighborsRegressor(), parameters)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_, '\n\n')

y_pred = grid_search.predict(X_test)
print_scores(y_test, y_pred)  # RMSE: 0.531750607527887

y_pred = grid_search.predict(X_train)
print_scores(y_train, y_pred)  # RMSE: 0.0
print('\n')


best_k = grid_search.best_params_["n_neighbors"]
best_weights = grid_search.best_params_["weights"]
bagged_knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_weights)

bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)
bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)
print_scores(y_test, y_pred)  # RMSE: 0.5296233305284472

y_pred = bagging_model.predict(X_train)
print_scores(y_train, y_pred)  # RMSE: 0.17819336310128217
