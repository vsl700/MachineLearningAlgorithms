from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
print(clf.fit(X, y))  # Prints the type of the fitted algorithm (SVC in this case)

X_new = [[2., 2.]]
y_new = clf.predict(X_new)
print(y_new)

print(clf.support_vectors_)  # Support vectors
print(clf.support_)  # Indices of support vectors
print(clf.n_support_)
