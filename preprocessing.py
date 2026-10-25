"""Numpy-only cleaning of the BRFSS 2015 features.

Steps, for each column:
    1. drop survey bookkeeping and raw answers that BRFSS already cleans elsewhere
    2. turn "don't know" / "refused" codes into missing, and "none" codes into 0
    3. drop the column if it is missing in too many training rows
    4. unordered categories: one 0/1 column per category
       other columns: fill missing with the median, add a 0/1 "was missing"
       column, and clip outliers
    5. standardize every feature

Every statistic is learned on the training rows only (fit_preprocessing) and
reused on any other rows (transform), so nothing leaks from the rows we
predict. In cross-validation, call fit_preprocessing on each training fold.

Usage:
    names = load_feature_names(data_path)
    params = fit_preprocessing(x_train, names)
    tx_train = transform(x_train, params)
    tx_test = transform(x_test, params)
"""

import os

import numpy as np

# Survey bookkeeping (dates, ids, phone screening, sampling weights) and
# questions about a child of the household rather than the respondent
ADMIN_COLUMNS = set(
    """
    FMONTH IDATE IMONTH IDAY IYEAR DISPCODE SEQNO _PSU CTELENUM PVTRESD1
    COLGHOUS STATERES CELLFON3 LADULT CTELNUM1 CELLFON2 CADULT PVTRESD2
    CCLGHOUS CSTATE LANDLINE NUMHHOL2 NUMPHON2 CPDEMO1 QSTVER QSTLANG _STSTR
    _STRWT _RAWRAKE _WT2RAKE _DUALUSE _DUALCOR _LLCPWT _CLLCPWT
    _CHISPNC _CRACE1 _CPRACE
    """.split()
)

# Raw answers in mixed units (e.g. 1xx = per day, 2xx = per week) or date
# formats. BRFSS provides cleaned versions of most of them (WTKG3, HTM4,
# DROCDY3_, FRUTDA1_, PAFREQ1_, STRFREQ_, ...)
SUPERSEDED_COLUMNS = set(
    """
    WEIGHT2 HEIGHT3 ALCDAY5 FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1
    EXRACT11 EXEROFT1 EXERHMM1 EXRACT21 EXEROFT2 EXERHMM2 STRENGTH
    FLSHTMY2 HIVTSTD3 BLDSUGAR FEETCHK2 LONGWTCH
    """.split()
)

# "Don't know" / "refused" codes become missing, "none" codes become 0.
# Which codes a column uses depends on how many digits its answers have,
# which we read from the column's largest value: 9, 99 or 999.
# That value is read on the training rows, so fit on the full training set or
# a CV fold of it: a small subsample (sub_sample=True) can miss the top code.
CODES_BY_MAX = {
    9: {7: np.nan, 9: np.nan},
    99: {77: np.nan, 99: np.nan, 88: 0},
    999: {777: np.nan, 999: np.nan, 888: 0},
}

# Columns whose codes differ from the rule above. Checked against the 2015
# codebook (data/dataset/codebook15_llcp.pdf) for every column kept by the
# full fit; some mostly-missing columns break the rule too (DIABAGE2 uses
# 98 for "don't know"), so check them before raising MAX_MISSING
SPECIAL_CODES = {
    "EMPLOY1": {9: np.nan},  # 7 = retired and 8 = unable to work are answers
    "_RACE": {9: np.nan},  # 7 = multiracial and 8 = Hispanic are answers
    "_AGEG5YR": {14: np.nan},  # 14 = age unknown, not "older than 80"
    "_AGE65YR": {3: np.nan},
    "SEATBELT": {7: np.nan, 8: np.nan, 9: np.nan},  # 8 = never drives or rides
    "MAXVO2_": {999: np.nan},
    "FC60_": {999: np.nan},
    "DROCDY3_": {900: np.nan},
    "_DRNKWEK": {99900: np.nan},
}

# Unordered categories: one 0/1 column per category instead of one number
ONE_HOT_COLUMNS = set(
    "_STATE MARITAL EMPLOY1 _RACE BPHIGH4 DIABETE3 RENTHOM1 PERSDOC2 MSCODE".split()
)

MAX_MISSING = 0.8  # drop a column missing in more than 80% of training rows
CLIP_PERCENTILES = (1, 99)
MIN_VALUES_TO_CLIP = 10  # yes/no and small ordinal answers are never clipped


def load_feature_names(data_path: str) -> list[str]:
    """Column names of x_train.csv without Id, in the order of load_csv_data."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def fit_preprocessing(
    x_train: np.ndarray, names: list[str], max_missing: float = MAX_MISSING
) -> dict:
    """
    Learn every cleaning statistic from the training rows.

    Args:
        x_train (np.array): raw training features (N, D) from load_csv_data
        names (list): the D column names, see load_feature_names
        max_missing (float, optional): drop columns missing more often than this

    Returns:
        params (dict): everything transform needs
    """
    if x_train.shape[1] != len(names):
        raise ValueError(f"x_train has {x_train.shape[1]} columns, got {len(names)} names")

    columns = []
    seen_missing_patterns = set()
    for index, name in enumerate(names):
        if name in ADMIN_COLUMNS or name in SUPERSEDED_COLUMNS:
            continue

        codes = special_codes(name, x_train[:, index])
        values = recode(x_train[:, index], codes)
        missing = np.isnan(values)
        observed = values[~missing]
        n_values = np.unique(observed).size
        if missing.mean() > max_missing or n_values < 2:
            continue

        info = {"index": index, "name": name, "codes": codes}
        if name in ONE_HOT_COLUMNS:
            # Missing counts as its own category, coded -1. The smallest
            # category gets no 0/1 column (it is the all-zeros case), so the
            # 0/1 columns never add up to the bias column.
            categories = np.unique(np.where(missing, -1, values))
            info["categories"] = categories[1:]
        else:
            info["median"] = np.median(observed)

            # Questions of a skipped module are all missing in the same rows:
            # one "was missing" column is enough for all of them
            pattern = missing.tobytes()
            info["add_flag"] = missing.any() and pattern not in seen_missing_patterns
            seen_missing_patterns.add(pattern)

            info["low"], info["high"] = -np.inf, np.inf
            if n_values > MIN_VALUES_TO_CLIP:
                info["low"], info["high"] = np.percentile(observed, CLIP_PERCENTILES)
        columns.append(info)

    features = build_features(x_train, columns)
    std = features.std(axis=0)
    return {
        "n_columns": len(names),
        "columns": columns,
        "mean": features.mean(axis=0),
        "std": np.where(std > 0, std, 1.0),
    }


def transform(x: np.ndarray, params: dict) -> np.ndarray:
    """
    Clean any rows (train, validation or test) with the training statistics.

    Args:
        x (np.array): raw features (N, D), same columns as at fit time
        params (dict): output of fit_preprocessing

    Returns:
        features (np.array): (N, D') standardized, no NaN, no bias column
    """
    if x.shape[1] != params["n_columns"]:
        raise ValueError(f"x has {x.shape[1]} columns, expected {params['n_columns']}")

    features = build_features(x, params["columns"])
    # In place on a fresh array: avoids two extra ~1 GB copies on the full data
    features -= params["mean"]
    features /= params["std"]
    return features


def special_codes(name: str, column: np.ndarray) -> dict:
    """Codes to replace in one column, from its name or its largest value."""
    if name in SPECIAL_CODES:
        return SPECIAL_CODES[name]
    observed = column[~np.isnan(column)]
    if observed.size == 0 or np.any(observed != np.round(observed)):
        return {}  # computed values such as servings per day, not answer codes
    return CODES_BY_MAX.get(observed.max(), {})


def recode(column: np.ndarray, codes: dict) -> np.ndarray:
    """Copy of the column with each code replaced by its value."""
    recoded = column.copy()
    for code, value in codes.items():
        recoded[column == code] = value
    return recoded


def build_features(x: np.ndarray, columns: list[dict]) -> np.ndarray:
    """Recode, fill, clip, flag and one-hot each kept column (not standardized)."""
    features = []
    for info in columns:
        values = recode(x[:, info["index"]], info["codes"])
        missing = np.isnan(values)
        if "categories" in info:
            # a missing or never-seen category gives all zeros
            features.extend(values == category for category in info["categories"])
        else:
            filled = np.where(missing, info["median"], values)
            features.append(np.clip(filled, info["low"], info["high"]))
            if info["add_flag"]:
                features.append(missing)
    # 0/1 columns become floats; copy=False avoids copying the result again
    return np.column_stack(features).astype(float, copy=False)
