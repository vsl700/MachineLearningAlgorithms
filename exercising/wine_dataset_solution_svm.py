from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from math import sqrt

data = datasets.load_wine()
print(data.feature_names, '\n')
print(data.target_names, '\n')
# print(data, '\n')

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.35, random_state=123)

print('Fitting model...')
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print('Done!\n')

y_pred = clf.predict(X_test)
print('Real:Prediction')
count = 0
for i in range(len(y_pred)):
    check = y_test[i] == y_pred[i]
    print(y_test[i], ':', y_pred[i], check)
    if check:
        count += 1
print('\n', count, '/', len(y_pred), ' correct', '\n', sep='')

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  # 0.9841269841269841
print("RMSE:", sqrt(metrics.mean_squared_error(y_test, y_pred)))  # 0.12598815766974242
