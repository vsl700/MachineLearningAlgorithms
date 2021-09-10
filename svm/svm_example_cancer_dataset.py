from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

cancer = datasets.load_breast_cancer()

print("Features:", cancer.feature_names, '\n')
print("Labels:", cancer.target_names, '\n')

print(cancer.data.shape, '\n')
print(cancer.data[:5], '\n')
print(cancer.target, '\n\n')


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

print('Fitting model...')
clf = svm.SVC(kernel='linear')  # Linear Kernel
clf.fit(X_train, y_train)
print('Done\n')

y_pred = clf.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))  # What percent are correct predictions
print('Root-mean Squared Error:', sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Precision:', metrics.precision_score(y_test, y_pred))  # What percent of all positive predictions ARE positives
print('Recall:', metrics.recall_score(y_test, y_pred))  # What percent of all positive actuals are captured

# Just made myself a graph
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_pred, s=50, cmap=cmap
)
f.colorbar(points)
plt.show()
