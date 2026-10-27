import sys
sys.path.insert(0, "..")   # also look in the parent folder (the repo root)
import numpy as np
import helpers
import preprocessing
import models


DATA_PATH = "./data"
SEED = 42


def f1_score(y_true, y_pred):
    """
    F1 score of binary predictions.

    Args:
        y_true (np.array): true labels in {0, 1}, shape (N,)
        y_pred (np.array): predicted labels in {0, 1}, shape (N,)

    Returns:
        f1 (float): F1 score, 0.0 if there is no true positive (also avoids dividing by zero)
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision*recall) / (precision+recall)

    return f1


def best_threshold(y_true, scores):
    """
    Find the threshold on the probability scores that gives the best F1.

    Args:
        y_true (np.array): true labels in {0, 1}, shape (N,)
        scores (np.array): predicted probabilities in [0, 1] (sigmoid of tx @ w), shape (N,)

    Returns:
        best_t (float): threshold with the best F1 (predict 1 if score >= best_t)
        best_f1 (float): F1 obtained with that threshold
    """
    thresholds = np.quantile(scores, np.linspace(0, 1, 101)) # different thresholds between 0 and 1 (steps of 0.01)

    f1_scores = []

    for t in thresholds:
        y_pred = (scores>=t).astype(int)
        f1 = f1_score(y_true, y_pred)

        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)

    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_t, best_f1


def subsample(x_train, y_train, size, rng):
    """
    Random subsample of the rows.

    Args:
        x_train (np.array): features, shape (N, D)
        y_train (np.array): labels, shape (N,)
        size (int): number of rows to keep
        rng (np.random.Generator): random generator, e.g. np.random.default_rng(42)

    Returns:
        x_sub (np.array): `size` distinct random rows of x_train, shape (size, D)
        y_sub (np.array): the matching labels, shape (size,)
    """
    idx = rng.choice(len(y_train), size=size, replace=False)  # `size` distinct row numbers in 0..N-1 - no replace avoid duplication

    return x_train[idx], y_train[idx]   # same idx for x and y: rows stay aligned


def split_data(x, y, ratio, rng):
    """
    Random split into a training part and a validation part.

    Args:
        x (np.array): features, shape (N, D)
        y (np.array): labels, shape (N,)
        ratio (float): share of the rows used for training (e.g. 0.8)
        rng (np.random.Generator): random generator

    Returns:
        x_tr, y_tr (np.array): training rows and labels (ratio * N rows)
        x_val, y_val (np.array): validation rows and labels (the rest)
    """
    cut = int(ratio*len(y))

    idx = rng.permutation(len(y))   # shuffled row numbers
    idx_tr = idx[:cut]
    idx_val = idx[cut:]

    x_tr, y_tr   = x[idx_tr], y[idx_tr]
    x_val, y_val = x[idx_val], y[idx_val]

    return x_tr, y_tr, x_val, y_val


def prepare_features(x_tr, x_val, names):
    """
    Fit the preprocessing on the training part only, transform both parts, add the bias.

    Args:
        x_tr (np.array): raw training features, shape (N_tr, 321)
        x_val (np.array): raw validation features, shape (N_val, 321)
        names (list): the 321 column names (preprocessing.load_feature_names)

    Returns:
        tx_tr (np.array): training matrix, standardized, bias column first
        tx_val (np.array): validation matrix, same columns, with the training statistics
    """
    params = preprocessing.fit_preprocessing(x_tr, names)
    tx_tr  = preprocessing.add_bias(preprocessing.transform(x_tr, params))
    tx_val = preprocessing.add_bias(preprocessing.transform(x_val, params))

    return tx_tr, tx_val


## CROSS VAL

def build_k_indices(y, k_fold, rng):
    """
    Stratified k-fold split of the row indices. taking care about the positives repartition

    Args:
        y (np.array): labels in {0, 1}, shape (N,)
        k_fold (int): number of folds
        rng (np.random.Generator): random generator

    Returns:
        k_indices (list of np.array): k_fold arrays of row indices, ~N/k_fold rows each,
                                      with the same share of positives in each fold
    """
    # 1 - row numbers of each class
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    # 2 - shuffle each class separately
    idx_pos = rng.permutation(idx_pos)
    idx_neg = rng.permutation(idx_neg)

    # 3 - cut each class into k_fold pieces (array_split keeps every row, even if N is not divisible)
    pos_parts = np.array_split(idx_pos, k_fold)
    neg_parts = np.array_split(idx_neg, k_fold)

    # 4 - fold i = piece i of the positives + piece i of the negatives
    k_indices = []
    for i in range(k_fold):
        folds = np.concatenate([pos_parts[i], neg_parts[i]])
        k_indices.append(folds)


    return k_indices



def cross_validate(x, y, names, k_indices, model, params, rng=None, verbose=True):
    """
    k-fold cross-validation of one model with one set of hyperparameters.

    Args:
        x (np.array): raw features, shape (N, 321)
        y (np.array): labels in {0, 1}, shape (N,)
        names (list): the 321 column names
        k_indices (list of np.array): folds from build_k_indices
        model (str): model name (see models.train_model)
        params (dict): hyperparameters of the model
        rng (np.random.Generator, optional): only used by mse_sgd
        verbose (bool): print the result of every fold

    Returns:
        f1_mean (float): mean validation F1 over the folds
        f1_std (float): standard deviation of the F1 over the folds
        t_mean (float): mean best threshold over the folds
    """

    f1s, thresholds = [], []

    for i in range(len(k_indices)):
        # 1 - validation = fold i, training = all the other folds + element definition
        idx_val = k_indices[i]
        idx_tr = np.concatenate([k_indices[j] for j in range(len(k_indices)) if j != i])

        x_tr, y_tr = x[idx_tr], y[idx_tr]
        x_val, y_val = x[idx_val], y[idx_val]

        # 2 - preprocessing: fit on the training folds only, transform both, add bias
        tx_tr, tx_val = prepare_features(x_tr, x_val, names)

        # 3 - train
        w, loss = models.train_model(model, y_tr, tx_tr, params, rng)

        # 4 - probabilities on the validation fold + best threshold
        scores = models.predict_scores(model, tx_val, w)
        best_t, best_f1 = best_threshold(y_val, scores)

        f1s.append(best_f1)
        thresholds.append(best_t)
        print(f"fold {i}: loss {loss:.4f}, threshold {best_t:.2f}, F1 {best_f1:.3f}")

    return np.mean(f1s), np.std(f1s), np.mean(thresholds)



def grid_search(x, y, names, k_indices, model, lambdas, gammas, iters_list, rng=None):
    """
    Grid search of the hyperparameters of one model with k-fold cross-validation.

    Args:
        x, y, names, k_indices: as in cross_validate
        model (str): model name (see models.train_model)
        lambdas, gammas, iters_list (list): values to test; parameters the model does not use are ignored
        rng (np.random.Generator, optional): only used by mse_sgd

    Returns:
        results (list of tuples): (lambda_, gamma, max_iters, f1_mean, f1_std, t_mean) for every combination
        best (tuple): the row of results with the highest f1_mean
    """
    results = []

    for lambda_ in lambdas:
        for gamma in gammas:
            for max_iters in iters_list:

                # 1 - evaluate this combination
                params = {"lambda_": lambda_, "gamma": gamma, "max_iters": max_iters}
                f1_mean, f1_std, t_mean = cross_validate(x, y, names, k_indices, model, params, rng=rng, verbose=False)

                # 2 - store it
                results.append((lambda_, gamma, max_iters, f1_mean, f1_std, t_mean))

                print(f"λ={lambda_:g}, γ={gamma:g}, iters={max_iters}: "
                      f"F1 {f1_mean:.3f} ± {f1_std:.3f}, threshold {t_mean:.2f}")

    # 3 - best combination = highest mean F1
    f1_means = np.array(results)[:, 3] # f1_means = [row[3] for row in results]
    best = results[np.argmax(f1_means)]

    return results, best



def compare_configs(x, y, names, k_indices, configs, seed=42):
    """
    Cross-validate a list of (model, params) configurations and return the best one.

    Args:
        x, y, names, k_indices: as in cross_validate
        configs (list of tuples): (model, params), e.g. [("ridge", {"lambda_": 1e-4}), ...]
        seed (int): seed of a fresh generator for each configuration (only used by mse_sgd)

    Returns:
        results (list of tuples): (model, params, f1_mean, f1_std, t_mean) for every configuration
        best (tuple): the row of results with the highest f1_mean
    """
    results = []
    for model, params in configs:
        f1_mean, f1_std, t_mean = cross_validate(x, y, names, k_indices, model, params,
                                                 rng=np.random.default_rng(seed), verbose=False)
        results.append((model, params, f1_mean, f1_std, t_mean))
        print(f"{model:14} {str(params):50} F1 {f1_mean:.3f} ± {f1_std:.3f}   threshold {t_mean:.3g}")

    best = max(results, key=lambda r: r[2])     # row with the highest f1_mean
    model, params, f1_mean, f1_std, t_mean = best
    print(f"\nbest: {model} {params} → F1 {f1_mean:.3f} ± {f1_std:.3f}, threshold {t_mean:.3g}")
    
    return results, best