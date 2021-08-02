import numpy as np
from sklearn.linear_model import LinearRegression


def PrintResponse(y_prediction):
    print('', "Predicted response:", y_prediction, sep='\n')


x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

print(x, y, sep='\n\n')

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print("Coefficient of determination:", r_sq)
print("Intercept (b0):", model.intercept_)
print("Slope (b1, b2):", model.coef_)

y_pred = model.predict(x)  # Basically each element of y is the sum of the predicted x values in each row
PrintResponse(y_pred)

# print('\n', model.coef_ * x)
y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
PrintResponse(y_pred)

print('\n')

x_new = np.arange(10).reshape(-1, 2)
print(x_new)

y_new = model.predict(x_new)
PrintResponse(y_new)
