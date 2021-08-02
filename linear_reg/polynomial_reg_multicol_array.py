import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
print(x_)

model = LinearRegression()
model.fit(x_, y)

r_sq = model.score(x_, y)
print("Coefficient of determination:", r_sq)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_, sep='\n')

y_pred = model.predict(x_)
print("Predictions:", y_pred, sep='\n')
