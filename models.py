
import numpy as np
import implementations




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
        return implementations.least_squares(y, tx)
    if model == "ridge":
        return implementations.ridge_regression(y, tx, params["lambda_"])
    if model == "mse_gd":
        return implementations.mean_squared_error_gd(y, tx, initial_w, params["max_iters"], params["gamma"])
    if model == "mse_sgd":
        return implementations.mean_squared_error_sgd(y, tx, initial_w, params["max_iters"], params["gamma"], rng=None)
    if model == "logistic":
        return implementations.logistic_regression(y, tx, initial_w, params["max_iters"], params["gamma"])
    if model == "reg_logistic":
        return implementations.reg_logistic_regression(y, tx, params["lambda_"], initial_w, params["max_iters"], params["gamma"])
    
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