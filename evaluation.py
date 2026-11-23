import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict

from auto import get_data_for_model_evaluation

x_data, y_data, x_train, x_test, y_train, y_test = get_data_for_model_evaluation()


def run_cross_validation(x_data, y_data, cv=4):
    lre = LinearRegression()

    Rcross = cross_val_score(lre, x_data[["horsepower"]], y_data, cv=cv)

    print(f"R² по фолдах: {Rcross}")
    print(f"Середнє R²: {Rcross.mean():.4f}")
    print(f"Стандартне відхилення: {Rcross.std():.4f}")

    return Rcross


def run_cross_validation_mse(x_data, y_data, cv=4):
    lre = LinearRegression()

    scores = cross_val_score(
        lre,
        x_data[["horsepower"]],
        y_data,
        cv=cv,
        scoring="neg_mean_squared_error"
    )
    mse = -scores

    print(f"MSE по фолдах: {mse}")
    print(f"Середнє MSE: {mse.mean():.2f}")

    return mse


def run_cross_val_predict(x_data, y_data, cv=4):
    lre = LinearRegression()

    yhat = cross_val_predict(lre, x_data[["horsepower"]], y_data, cv=cv)

    print(f"Перші 5 передбачень: {yhat[:5]}")

    return yhat


if __name__ == "__main__":
    run_cross_validation(x_data, y_data)
    run_cross_validation_mse(x_data, y_data)
    run_cross_val_predict(x_data, y_data)