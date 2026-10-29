"""Feature expansion on top of the preprocessing: interactions and polynomial terms.
"""

import numpy as np


def combinations(indices, d):
    """
    Every group of d distinct elements of `indices`, keeping their order: e.g. [1, 2, 3], d = 2 -> (1, 2), (1, 3), (2, 3).

    Args:
        indices (np.array or list): elements to combine, e.g. column indices
        d (int): size of each group

    Returns:
        groups (list of tuples): the C(len(indices), d) groups
    """
    if d == 0:
        return [()]
    groups = []
    for i in range(len(indices)):
        for rest in combinations(
            indices[i + 1 :], d - 1
        ):  # the other elements come after position i
            groups.append((indices[i],) + rest)
    return groups


def feature_names(params):
    """Names of the columns of add_bias(transform(x, params)), in the same order."""
    out = ["bias"]
    for info in params["columns"]:
        if "categories" in info:
            out += [
                f"{info['name']}={c:g}" for c in info["categories"]
            ]  # one 0/1 column per category
        else:
            out.append(info["name"])
            if info["add_flag"]:
                out.append(f"{info['name']}_missing")
    return out


def pick_top(tx_tr, w, top, max_corr=0.9):
    """
    The `top` largest-weight features, skipping near-duplicates of a feature already picked.

    Args:
        tx_tr (np.array): standardized training matrix, bias column first
        w (np.array): weights of a fit on tx_tr, shape (D,)
        top (int): number of features to pick
        max_corr (float): skip a feature whose |correlation| with a picked one is above this (1.0 = keep all)

    Returns:
        picked (np.array): column indices of the picked features, strongest first
    """
    n = len(tx_tr)
    picked = []
    for j in 1 + np.argsort(-np.abs(w[1:])):  # strongest first, bias excluded
        # the columns are standardized, so their correlation is the mean of their product
        if all(abs(tx_tr[:, j] @ tx_tr[:, k]) / n <= max_corr for k in picked):
            picked.append(j)
        if len(picked) == top:
            break
    return np.array(picked)


def add_interactions(tx_tr, tx_val, w, degree=2, top=30, max_corr=0.9, max_gb=3.0):
    """
    Products of 2..degree distinct features among the `top` strongest ones, standardized with training stats.

    Args:
        tx_tr, tx_val (np.array): standardized matrices, bias column first
        w (np.array): weights of a fit on tx_tr, used to rank the features
        degree (int): 1 = unchanged, 2 = + pairs, 3 = + pairs and triples
        top (int): number of strongest features combined (see pick_top)
        max_corr (float): near-duplicates above this |correlation| are skipped (e.g. MAXVO2_ vs FC60_)
        max_gb (float): refuse to build a training matrix bigger than this (8 GB of RAM)

    Returns:
        tx_tr, tx_val (np.array): new matrices with C(top, 2) [+ C(top, 3)] product columns appended
    """
    if degree < 2:
        return tx_tr, tx_val
    best = pick_top(tx_tr, w, top, max_corr)
    combos = [c for d in range(2, degree + 1) for c in combinations(best, d)]
    gb = 8 * len(tx_tr) * (tx_tr.shape[1] + len(combos)) / 1e9
    if gb > max_gb:
        raise MemoryError(
            f"{len(combos)} products -> {gb:.1f} GB training matrix: lower top or degree"
        )
    p_tr = np.column_stack([np.prod(tx_tr[:, list(c)], axis=1) for c in combos])
    p_val = np.column_stack([np.prod(tx_val[:, list(c)], axis=1) for c in combos])
    mean, std = p_tr.mean(0), p_tr.std(0)
    std[std == 0] = 1
    return np.hstack([tx_tr, (p_tr - mean) / std]), np.hstack(
        [tx_val, (p_val - mean) / std]
    )


def numerical_columns(tx_tr):
    """
    Columns with more than two distinct values (age, BMI, days of poor health...).

    0/1 columns (one-hot, "was missing" flags, yes/no answers) and the bias are excluded:
    their powers are the column itself (after standardization, a linear function of it).

    Args:
        tx_tr (np.array): standardized training matrix, bias column first

    Returns:
        cols (np.array): indices of the numerical columns
    """
    lo, hi = tx_tr.min(axis=0), tx_tr.max(axis=0)
    binary = ((tx_tr == lo) | (tx_tr == hi)).all(
        axis=0
    )  # only two values (or constant)
    return np.flatnonzero(~binary)


def add_polynomial(tx_tr, tx_val, degree=2, w=None, top=None):
    """
    Powers 2..degree of the numerical columns, standardized with training stats.

    x^d is only built for a column with more than d distinct values: otherwise it is a combination
    of 1, x, ..., x^(d-1) (e.g. x³ of a 3-level answer), so it adds nothing and would make the
    normal equations of least_squares singular.

    Args:
        tx_tr, tx_val (np.array): standardized matrices, bias column first
        degree (int): 1 = unchanged, 2 = + x², 3 = + x² and x³
        w (np.array, optional): weights of a fit on tx_tr, needed only when `top` is set
        top (int, optional): None = every numerical column, else only the `top` largest-|w| ones

    Returns:
        tx_tr, tx_val (np.array): new matrices with the power columns appended
    """
    if degree < 2:
        return tx_tr, tx_val

    # 1 - which columns get powers: the numerical ones, or only the `top` strongest of them
    cols = numerical_columns(tx_tr)
    if top is not None:
        if w is None:
            raise ValueError(
                "add_polynomial: w is needed to pick the `top` strongest columns"
            )
        cols = cols[np.argsort(-np.abs(w[cols]))][:top]

    # 2 - (column, power) pairs, skipping the powers that only repeat information
    n_values = [np.unique(tx_tr[:, j]).size for j in cols]
    powers = [
        (j, d) for d in range(2, degree + 1) for j, n in zip(cols, n_values) if n > d
    ]
    if not powers:
        return tx_tr, tx_val

    # 3 - powers of the standardized values, then standardized again with the training stats
    p_tr = np.column_stack([tx_tr[:, j] ** d for j, d in powers])
    p_val = np.column_stack([tx_val[:, j] ** d for j, d in powers])
    mean, std = p_tr.mean(0), p_tr.std(0)
    std[std == 0] = 1
    return np.hstack([tx_tr, (p_tr - mean) / std]), np.hstack(
        [tx_val, (p_val - mean) / std]
    )
