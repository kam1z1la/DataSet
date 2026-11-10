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