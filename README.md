# ML Project 1: predicting heart disease (BRFSS 2015)

EPFL CS-433 Machine Learning, project 1.
The goal is to predict whether a person has coronary heart disease from their answers to the
BRFSS 2015 health survey. The data has 328,135 training rows, 321 features and about 9% positives.
Submissions are scored with the **F1 score** on AIcrowd.

**Final result: F1 = 0.443 on AIcrowd** (regularized logistic regression + x² and interactions, `run.py`).

Everything is written with **numpy only**. Matplotlib is used only in the notebooks, for plots.

## Project structure

```
ml_project1_jab/
├── data/                     # the CSV files go here (not tracked by git)
├── submissions/              # prediction files written by run.py
├── notebooks/
│   ├── experiments_log.md    # every experiment and its result (rows #1–102)
│   └── *.ipynb               # personal exploration notebooks
├── experiments.ipynb         # cross-validation experiments (models, hyperparameters, feature expansion)
├── helpers.py                # load the CSVs, write the submission file (given by the course)
├── preprocessing.py          # cleaning of the 321 raw features
├── features_expansion.py     # polynomial terms and interactions
├── implementations.py        # the 6 required models + train_model / predict_scores
├── cross_validation.py       # F1, threshold, stratified k-fold, grid searches
└── run.py                    # trains the final model and writes the submission
```

## Setup

1. Install Python 3 with numpy.
2. Put `x_train.csv`, `y_train.csv` and `x_test.csv` from AIcrowd in `data/`.

## Reproduce the submission

From the project root:

```bash
python run.py
```

This takes about 5 minutes. Reading the CSVs takes about 2 minutes of that.
The file is written to the `OUTPUT` path set at the top of `run.py`, in `submissions/`.
The model, its parameters and the threshold are also set at the top of `run.py`:

```python
MODEL = "reg_logistic"
PARAMS = {"lambda_": 1e-3, "gamma": 0.1, "max_iters": 1000,
          "poly_degree": 2, "poly_top": None, "inter_degree": 2, "inter_top": 20, "max_corr": 0.9}
THRESHOLD = None          # None = choose the threshold that maximizes F1 on the training rows
```

## Pipeline

1. **Preprocessing** (`preprocessing.py`). For each column:
   - drop survey bookkeeping columns and raw answers that BRFSS already cleans elsewhere;
   - turn "don't know" / "refused" codes into missing values, and "none" codes into 0;
   - drop columns that are missing too often;
   - one-hot encode unordered categories;
   - fill other missing values with the median and add a 0/1 "was missing" column;
   - clip outliers and standardize.

   Every statistic is learned on the training rows only (`fit_preprocessing`) and reused on the
   other rows (`transform`).
2. **Feature expansion** (`features_expansion.py`, optional, set by the `PARAMS` keys):
   - `poly_degree`, `poly_top`: x², x³ of the numerical columns, either all of them or only the strongest ones;
   - `inter_degree`, `inter_top`, `max_corr`: products of pairs (or triples) of the strongest features.
     The strongest features are ranked by the weights of a first fit, and near-duplicate columns are skipped.
3. **Model** (`implementations.py`). `train_model(model, y, tx, params)` trains one of the 6 models:
   `least_squares`, `ridge`, `mse_gd`, `mse_sgd`, `logistic`, `reg_logistic`.
4. **Threshold**. The model outputs a probability, and a row is predicted positive if the probability is
   above a threshold. Because only ~9% of the rows are positive, the threshold that maximizes F1 is
   about 0.22, not 0.5. It is chosen on the training rows (`best_threshold`) and then applied to the test rows.

## Experiments

`experiments.ipynb` runs stratified 5-fold cross-validation on a subsample (`N_SUB` rows):

- `cross_validate`: one model with one set of parameters;
- `grid_search`: λ, γ and number of iterations;
- `compare_configs`: any list of (model, params);
- `grid_search_expansion`: polynomial degree and interactions.

In cross-validation, preprocessing, feature expansion and the threshold are all fitted on the training
folds only, so the validation F1 is an honest estimate.

Every run is logged in [`notebooks/experiments_log.md`](notebooks/experiments_log.md). Main results:

| Finding | F1 (5-fold CV) |
|---|---|
| Logistic / regularized logistic regression | **0.429** ± 0.012 |
| Least squares, ridge, MSE gradient descent | 0.421–0.423 |
| MSE SGD | ≤ 0.37, unstable |
| Same logistic model on the full 328k rows | 0.429 ± 0.002 |
| + x² and pairs of the 20 strongest features (final `run.py`) | 0.429 ± 0.009 (AIcrowd: **0.443**) |

- Once gradient descent has converged, the number of iterations, γ and λ (up to 1e-2) do not change the F1.
- Polynomial terms and interactions do not beat the plain model beyond noise. Large expansions overfit.
