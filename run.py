import os
import numpy as np
import helpers
import preprocessing
import implementations
import cross_validation


DATA_PATH = "./data"
SEED = 42


# TODO : Hyperparameters
MODEL = "reg_logistic"
PARAMS = {
    "lambda_": 1e-3,
    "gamma": 0.1,
    "max_iters": 1000,
    "poly_degree": 2,
    "poly_top": None,
    "inter_degree": 2,
    "inter_top": 20,
    "max_corr": 0.9,
}  # x² of all numerical columns + pairs of the 20 strongest features
THRESHOLD = None
OUTPUT = "submissions/submission_test_poly2_inter20.csv"


## DATA LOADING
x_train, x_test, y_train, _, test_ids = helpers.load_csv_data(DATA_PATH)
y = (y_train == 1).astype(int)

## PREPROCESSING: fit on all of x_train, transform both
names = preprocessing.load_feature_names(DATA_PATH)
params_prep = preprocessing.fit_preprocessing(x_train, names)
tx_tr = preprocessing.add_bias(preprocessing.transform(x_train, params_prep))
tx_te = preprocessing.add_bias(preprocessing.transform(x_test, params_prep))

## FEATURE EXPANSION: the polynomial / interaction columns asked for in PARAMS (training statistics only)
tx_tr, tx_te = cross_validation.expand_features(
    tx_tr, tx_te, y, MODEL, PARAMS, rng=np.random.default_rng(SEED)
)

## TRAINING
w, loss = implementations.train_model(
    MODEL, y, tx_tr, PARAMS, rng=np.random.default_rng(SEED)
)

## THRESHOLD: chosen on the training rows
if THRESHOLD is None:
    THRESHOLD, _ = cross_validation.best_threshold(
        y, implementations.predict_scores(MODEL, tx_tr, w)
    )

## PREDICTION
scores = implementations.predict_scores(MODEL, tx_te, w)
y_pred = (scores >= THRESHOLD).astype(int)
y_pred[y_pred == 0] = -1

## SUBMISSION
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
helpers.create_csv_submission(test_ids, y_pred, OUTPUT)

## CHECKS
print("training loss:", loss, " threshold:", THRESHOLD)
print(y_pred.shape, np.unique(y_pred, return_counts=True))
print("share predicted positive:", np.mean(y_pred == 1))
