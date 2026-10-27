import os
import numpy as np
import helpers
import preprocessing
import models



DATA_PATH = "./data"
SEED = 42


# TODO : Hyperparameters
MODEL = "reg_logistic"
PARAMS = {"lambda_": 1e-4, "gamma": 0.1, "max_iters": 1000}
THRESHOLD = 0.20          # mean best threshold from the CV, in the units of predict_scores
OUTPUT = "submissions/submission.csv"



## DATA LOADING
x_train, x_test, y_train, _, test_ids = helpers.load_csv_data(DATA_PATH)
y = (y_train == 1).astype(int)

## PREPROCESSING: fit on all of x_train, transform both
names = preprocessing.load_feature_names(DATA_PATH)
params_prep = preprocessing.fit_preprocessing(x_train, names)
tx_tr = preprocessing.add_bias(preprocessing.transform(x_train, params_prep))
tx_te = preprocessing.add_bias(preprocessing.transform(x_test, params_prep))

## TRAINING
w, loss = models.train_model(MODEL, y, tx_tr, PARAMS, rng=np.random.default_rng(SEED))

## PREDICTION
scores = models.predict_scores(MODEL, tx_te, w)
y_pred = (scores >= THRESHOLD).astype(int)
y_pred[y_pred == 0] = -1

## SUBMISSION
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
helpers.create_csv_submission(test_ids, y_pred, OUTPUT)

## CHECKS
print("training loss:", loss)
print(y_pred.shape, np.unique(y_pred, return_counts=True))
print("share predicted positive:", np.mean(y_pred == 1))