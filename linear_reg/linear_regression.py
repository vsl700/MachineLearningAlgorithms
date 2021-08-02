import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

print(x)
print(y)

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)
print("Coefficient of determination:", r_sq)

print("Intercept (b0, or y when x is 0):", model.intercept_)
print("Slope (b1, or how much y increases by when x increases by 1):", model.coef_)

"""
Sum of X = 180
Sum of Y = 131
Mean X = 30
Mean Y = 21.8333
Sum of squares (SSX) = 1750
Sum of products (SP) = 945

Regression Equation = ŷ = bX + a

b = SP/SSX = 945/1750 = 0.54

a = MY - bMX = 21.83 - (0.54*30) = 5.63333

ŷ = 0.54X + 5.63333
"""

y_pred = model.predict(x)
print("Predicted response: ", y_pred, sep='\n')
print("\n-------------------------------------------------------------------\n")

x_new = np.arange(5).reshape(-1, 1)
print(x_new)

y_new = model.predict(x_new)
print("New predicted response:", y_new, sep='\n')
