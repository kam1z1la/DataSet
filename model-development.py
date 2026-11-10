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