from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from math import sqrt


def print_scores(y_true, y_pr):
    print('RMSE:', sqrt(metrics.mean_squared_error(y_true, y_pr)))
    print('Accuracy:', metrics.accuracy_score(y_true, y_pr))


data = datasets.load_wine()
print(len(data.data[:, 0]))  # 178

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.35, random_state=123)

clf = KNeighborsClassifier(n_neighbors=3)  # train error is 0 with distance weights
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print_scores(y_test, y_pred)  # RMSE: 0.7559289460184544, Acc: 0.6666666666666666

y_pred = clf.predict(X_train)
print_scores(y_train, y_pred)  # RMSE: 0.5021692075819447, Acc: 0.8521739130434782
print('\n')

# Improving kNN
parameters = {"n_neighbors": range(1, 50),
              "weights": ["uniform", "distance"]}
grid_search = GridSearchCV(KNeighborsClassifier(), parameters)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_, '\n\n')

y_pred = grid_search.predict(X_test)
print_scores(y_test, y_pred)  # RMSE: 0.6546536707079771, Acc: 0.7142857142857143

y_pred = grid_search.predict(X_train)
print_scores(y_train, y_pred)  # RMSE: 0.0, Acc: 1.0
print('\n')


best_k = grid_search.best_params_["n_neighbors"]
best_weights = grid_search.best_params_["weights"]
bagged_knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)

bagging_model = BaggingClassifier(bagged_knn, n_estimators=100)
bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)
print_scores(y_test, y_pred)  # RMSE: 0.6546536707079771, Acc: 0.7142857142857143

y_pred = bagging_model.predict(X_train)
print_scores(y_train, y_pred)  # RMSE: 0.0, Acc: 1.0
