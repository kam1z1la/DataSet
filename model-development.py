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

def plot_regression(x_name, y_name="price"):
    plt.figure(figsize=(12, 10))
    sns.regplot(x=x_name, y=y_name, data=df)
    plt.ylim(0,)
    plt.show()


def plot_residuals(x_name, y_name="price"):
    plt.figure(figsize=(12, 10))
    sns.residplot(x=df[x_name], y=df[y_name])
    plt.show()


X = df[["highway-mpg"]]
Y = df["price"]

lm = LinearRegression()
lm.fit(X, Y)

print("Intercept:", lm.intercept_)
print("Coefficient:", lm.coef_[0])
print(f"price = {lm.intercept_:.2f} + {lm.coef_[0]:.2f} * highway-mpg")

Yhat = lm.predict(X)
print("First 5 predictions:", Yhat[:5])

plot_regression("highway-mpg")
plot_residuals("highway-mpg")

print("R^2:", r2_score(Y, Yhat))
print("MSE:", mean_squared_error(Y, Yhat))
print("R^2 via score:", lm.score(X, Y))

def plot_regression(x_name, y_name="price"):
    plt.figure(figsize=(12, 10))
    sns.regplot(x=x_name, y=y_name, data=df)
    plt.ylim(0,)
    plt.show()


def plot_residuals(x_name, y_name="price"):
    plt.figure(figsize=(12, 10))
    sns.residplot(x=df[x_name], y=df[y_name])
    plt.show()


X = df[["highway-mpg"]]
Y = df["price"]

lm = LinearRegression()
lm.fit(X, Y)

print("Intercept:", lm.intercept_)
print("Coefficient:", lm.coef_[0])
print(f"price = {lm.intercept_:.2f} + {lm.coef_[0]:.2f} * highway-mpg")

Yhat = lm.predict(X)
print("First 5 predictions:", Yhat[:5])

plot_regression("highway-mpg")
plot_residuals("highway-mpg")

print("R^2:", r2_score(Y, Yhat))
print("MSE:", mean_squared_error(Y, Yhat))
print("R^2 via score:", lm.score(X, Y))


def plot_regression(x_name, y_name="price"):
    plt.figure(figsize=(12, 10))
    sns.regplot(x=x_name, y=y_name, data=df)
    plt.ylim(0,)
    plt.show()


def plot_residuals(x_name, y_name="price"):
    plt.figure(figsize=(12, 10))
    sns.residplot(x=df[x_name], y=df[y_name])
    plt.show()


X = df[["highway-mpg"]]
Y = df["price"]

lm = LinearRegression()
lm.fit(X, Y)

print("Intercept:", lm.intercept_)
print("Coefficient:", lm.coef_[0])
print(f"price = {lm.intercept_:.2f} + {lm.coef_[0]:.2f} * highway-mpg")

Yhat = lm.predict(X)
print("First 5 predictions:", Yhat[:5])

plot_regression("highway-mpg")
plot_residuals("highway-mpg")

print("R^2:", r2_score(Y, Yhat))
print("MSE:", mean_squared_error(Y, Yhat))
print("R^2 via score:", lm.score(X, Y))

X = df[["highway-mpg"]]
Y = df["price"]

lm = LinearRegression()
lm.fit(X, Y)

print("Intercept:", lm.intercept_)
print("Coefficient:", lm.coef_[0])

X = df[["highway-mpg"]]
Y = df["price"]

lm = LinearRegression()
lm.fit(X, Y)

Yhat = lm.predict(X)

print("Intercept:", lm.intercept_)
print("Coefficient:", lm.coef_[0])
print("First 5 predictions:", Yhat[:5])

X = df[["highway-mpg"]]
Y = df["price"]

lm = LinearRegression()
lm.fit(X, Y)

Yhat = lm.predict(X)

print("Intercept:", lm.intercept_)
print("Coefficient:", lm.coef_[0])
print(f"price = {lm.intercept_:.2f} + {lm.coef_[0]:.2f} * highway-mpg")
print("First 5 predictions:", Yhat[:5])