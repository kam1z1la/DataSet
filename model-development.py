import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from auto import get_dataset_for_regression

df = get_dataset_for_regression()

X = df[["highway-mpg"]]
Y = df["price"]

lm = LinearRegression()
lm.fit(X, Y)

Yhat = lm.predict(X)

print("Intercept:", lm.intercept_)
print("Coefficient:", lm.coef_[0])
print(f"price = {lm.intercept_:.2f} + {lm.coef_[0]:.2f} * highway-mpg")
print("First 5 predictions:", Yhat[:5])

plt.figure(figsize=(12, 10))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.show()

plt.figure(figsize=(12, 10))
sns.residplot(x=df["highway-mpg"], y=df["price"])
plt.show()

r2 = r2_score(Y, Yhat)
mse = mean_squared_error(Y, Yhat)

print("R^2:", r2)
print("R^2 (lm.score):", lm.score(X, Y))
print("MSE:", mse)

features = ["horsepower", "curb-weight", "engine-size", "highway-mpg"]
Z = df[features]

lm_multi = LinearRegression()
lm_multi.fit(Z, Y)

Y_hat_multi = lm_multi.predict(Z)

print("MLR intercept:", lm_multi.intercept_)
print("MLR coefficients:", dict(zip(features, lm_multi.coef_)))
print("MLR R^2:", r2_score(Y, Y_hat_multi))
print("MLR MSE:", mean_squared_error(Y, Y_hat_multi))

plt.figure(figsize=(12, 10))
ax1 = sns.kdeplot(Y, color="r", label="Actual")
sns.kdeplot(Y_hat_multi, color="b", label="Fitted", ax=ax1)
plt.legend()
plt.show()

x = df["highway-mpg"]
y = df["price"]

f = np.polyfit(x, y, 3)
p = np.poly1d(f)

print("Polynomial (deg=3):")
print(p)

x_new = np.linspace(x.min(), x.max(), 100)

plt.figure(figsize=(12, 10))
plt.plot(x, y, ".", x_new, p(x_new), "-")
plt.xlabel("highway-mpg")
plt.ylabel("price")
plt.show()

print("Polynomial R^2:", r2_score(y, p(x)))
print("Polynomial MSE:", mean_squared_error(y, p(x)))