import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from auto import get_data_for_model_evaluation
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

from auto import get_data_for_model_evaluation

x_data, y_data, x_train, x_test, y_train, y_test = get_data_for_model_evaluation()

RIDGE_FEATURES = [
    "horsepower",
    "curb-weight",
    "engine-size",
    "highway-mpg",
    "normalized-losses",
    "symboling"
]

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


def run_polynomial_degrees(x_train, x_test, y_train, y_test, max_degree=5):
    degrees = list(range(1, max_degree + 1))
    r2_train = []
    r2_test = []

    for degree in degrees:
        pr = PolynomialFeatures(degree=degree)

        x_train_pr = pr.fit_transform(x_train[["horsepower"]])
        x_test_pr = pr.transform(x_test[["horsepower"]])

        lr = LinearRegression()
        lr.fit(x_train_pr, y_train)

        r2_train.append(lr.score(x_train_pr, y_train))
        r2_test.append(lr.score(x_test_pr, y_test))

        print(
            f"Степінь {degree}: "
            f"R² train = {r2_train[-1]:.4f}, "
            f"R² test = {r2_test[-1]:.4f}"
        )

    best_degree = degrees[int(np.argmax(r2_test))]
    print(f"Оптимальний степінь за R² на test: {best_degree}")

    return degrees, r2_train, r2_test, best_degree


def plot_polynomial_degrees(degrees, r2_train, r2_test):
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, r2_train, marker="o", label="Train R²")
    plt.plot(degrees, r2_test, marker="o", label="Test R²")
    plt.xlabel("Степінь полінома")
    plt.ylabel("R²")
    plt.title("Перенавчання: R² залежно від степеня полінома")
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True)
    plt.show()


def prepare_ridge_features(x_train, x_test, degree=2):
    pr = PolynomialFeatures(degree=degree)

    x_train_pr = pr.fit_transform(x_train[RIDGE_FEATURES])
    x_test_pr = pr.transform(x_test[RIDGE_FEATURES])

    print(f"Ознак після PolynomialFeatures: {x_train_pr.shape[1]}")

    return x_train_pr, x_test_pr


def run_ridge(x_train_pr, x_test_pr, y_train, y_test, alpha=1):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(x_train_pr, y_train)

    r2_train = ridge_model.score(x_train_pr, y_train)
    r2_test = ridge_model.score(x_test_pr, y_test)

    print(f"Ridge (alpha={alpha}): R² train = {r2_train:.4f}, R² test = {r2_test:.4f}")

    return ridge_model


def run_ridge_alpha_search(x_train_pr, x_test_pr, y_train, y_test, alphas=None):
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    r2_train = []
    r2_test = []

    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(x_train_pr, y_train)

        r2_train.append(ridge_model.score(x_train_pr, y_train))
        r2_test.append(ridge_model.score(x_test_pr, y_test))

        print(
            f"alpha = {alpha}: "
            f"R² train = {r2_train[-1]:.4f}, "
            f"R² test = {r2_test[-1]:.4f}"
        )

    best_alpha = alphas[int(np.argmax(r2_test))]
    print(f"Найкращий alpha за R² на test: {best_alpha}")

    return alphas, r2_train, r2_test, best_alpha


def plot_ridge_alpha_search(alphas, r2_train, r2_test):
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, r2_train, marker="o", label="Train R²")
    plt.plot(alphas, r2_test, marker="o", label="Test R²")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("R²")
    plt.title("Ridge: R² залежно від alpha")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_grid_search(x_train_pr, x_test_pr, y_train, y_test, cv=4):
    parameters1 = [{"alpha": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]

    RR = Ridge()
    Grid1 = GridSearchCV(RR, parameters1, cv=cv)
    Grid1.fit(x_train_pr, y_train)

    best_model = Grid1.best_estimator_

    print(f"Кращі параметри: {Grid1.best_params_}")
    print(f"Найкращий середній R² на крос-валідації: {Grid1.best_score_:.4f}")
    print(f"R² кращої моделі на train: {best_model.score(x_train_pr, y_train):.4f}")
    print(f"R² кращої моделі на test: {best_model.score(x_test_pr, y_test):.4f}")

    return Grid1, best_model


def print_grid_search_results(Grid1):
    scores = Grid1.cv_results_
    for params, mean_score, std_score in zip(
        scores["params"],
        scores["mean_test_score"],
        scores["std_test_score"]
    ):
        print(f"{params}: середнє R² = {mean_score:.4f}, std = {std_score:.4f}")

if __name__ == "__main__":
    run_cross_validation(x_data, y_data)
    run_cross_validation_mse(x_data, y_data)
    run_cross_val_predict(x_data, y_data)

    degrees, r2_train, r2_test, best_degree = run_polynomial_degrees(
        x_train, x_test, y_train, y_test
    )
    plot_polynomial_degrees(degrees, r2_train, r2_test)

    x_train_pr, x_test_pr = prepare_ridge_features(x_train, x_test)
    run_ridge(x_train_pr, x_test_pr, y_train, y_test, alpha=1)

    alphas, ridge_r2_train, ridge_r2_test, best_alpha = run_ridge_alpha_search(
        x_train_pr, x_test_pr, y_train, y_test
    )
    plot_ridge_alpha_search(alphas, ridge_r2_train, ridge_r2_test)

    Grid1, best_model = run_grid_search(x_train_pr, x_test_pr, y_train, y_test)
    print_grid_search_results(Grid1)