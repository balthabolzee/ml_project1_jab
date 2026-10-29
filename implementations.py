import numpy as np
from helpers import batch_iter


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with gradient descent. Returns (w, loss)."""

    w = initial_w

    for n_iter in range(max_iters):

        e = y - tx @ w
        grad_mse = -1 / len(e) * tx.T @ e
        w = w - gamma * grad_mse

    e = y - tx @ w
    loss = 1 / (2 * len(e)) * e.T @ e

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, rng=None):
    """Linear regression with SGD (batch size 1). Returns (w, loss)."""

    if rng is None:
        rng = np.random.default_rng()  # created once, before the loop

    w = initial_w

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=1, num_batches=1, rng=rng
        ):

            stochastic_e = y_batch - tx_batch @ w
            grad_mse = -1 / len(stochastic_e) * tx_batch.T @ stochastic_e

            w = w - gamma * grad_mse

    e = y - tx @ w
    loss = 1 / (2 * len(e)) * e.T @ e

    return w, loss


def least_squares(y, tx):
    """Least squares via normal equations (no np.linalg.lstsq)."""

    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    e = y - tx.dot(w)
    mse = 1 / 2 * np.mean(e**2)
    loss = mse

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression via normal equations. Loss without penalty."""

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    e = y - tx.dot(w)
    mse = 1 / 2 * np.mean(e**2)
    loss = mse

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent. Returns (w, loss)."""

    w = initial_w.copy()
    n = len(y)

    for _ in range(max_iters):
        z = tx @ w
        sigmoid = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        gradient = tx.T @ (sigmoid - y) / n
        w = w - gamma * gradient

    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression. Loss without penalty."""

    w = initial_w.copy()
    n = len(y)

    for _ in range(max_iters):
        z = tx @ w
        sigmoid = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        gradient = tx.T @ (sigmoid - y) / n
        gradient = gradient + 2 * lambda_ * w

        w = w - gamma * gradient

    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)

    return w, loss


def train_model(model, y, tx, params, rng=None):
    """
    Train one of the 6 models of implementations.py.

    Args:
        model (str): "least_squares", "ridge", "mse_gd", "mse_sgd", "logistic" or "reg_logistic"
        y (np.array): labels in {0, 1}, shape (N,)
        tx (np.array): features with bias column, shape (N, D)
        params (dict): hyperparameters used by the model, e.g. {"lambda_": 1e-4, "gamma": 0.1, "max_iters": 500}
        rng (np.random.Generator, optional): only used by mse_sgd

    Returns:
        w (np.array): trained weights, shape (D,)
        loss (float): training loss returned by the model
    """
    initial_w = np.zeros(tx.shape[1])

    if model == "least_squares":
        return least_squares(y, tx)
    if model == "ridge":
        return ridge_regression(y, tx, params["lambda_"])
    if model == "mse_gd":
        return mean_squared_error_gd(
            y, tx, initial_w, params["max_iters"], params["gamma"]
        )
    if model == "mse_sgd":
        return mean_squared_error_sgd(
            y, tx, initial_w, params["max_iters"], params["gamma"], rng=rng
        )
    if model == "logistic":
        return logistic_regression(
            y, tx, initial_w, params["max_iters"], params["gamma"]
        )
    if model == "reg_logistic":
        return reg_logistic_regression(
            y, tx, params["lambda_"], initial_w, params["max_iters"], params["gamma"]
        )

    raise ValueError(f"unknown model: {model}")


def predict_scores(model, tx, w):
    """
    Scores used to rank the rows: probabilities for logistic models, raw linear scores otherwise.

    Args:
        model (str): model name (see train_model)
        tx (np.array): features with bias column, shape (N, D)
        w (np.array): trained weights, shape (D,)

    Returns:
        scores (np.array): shape (N,), higher = more likely to be positive
    """
    z = tx @ w
    if model in ("logistic", "reg_logistic"):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    return z
