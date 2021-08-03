import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def print_regression_data():
    print('------------------------------------------------------')

    r_sq = model.score(x_, y)
    print("Coefficient of determination:", r_sq)
    print("Intercept (b0):", model.intercept_)
    print("Coefficients:", model.coef_)

    print('------------------------------------------------------\n')


def print_prediction():
    y_pred = model.predict(x_)
    print("Predicted response:", y_pred, '\n', sep='\n')


x = np.array([5, 15, 25, 35, 45, 55]).reshape(-1, 1)  # The input needs to be a two-dimensional array!
y = np.array([15, 11, 2, 8, 25, 32])

x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
print(x_)

model = LinearRegression()
model.fit(x_, y)

print_regression_data()

print_prediction()


x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
print(x_)

model = LinearRegression(fit_intercept=False)
model.fit(x_, y)

print_regression_data()

print_prediction()
