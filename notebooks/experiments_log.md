# Experiment log

| #  | Date  | Model        | Params (λ, γ, iters) | Setup (data, validation) | Seed | Train loss | Threshold | F1 (mean ± std) | AIcrowd F1 | Notes                                                        |
|----|-------|--------------|----------------------|--------------------------|------|------------|-----------|-----------------|------------|--------------------------------------------------------------|
| 1  | 26.10 | reg_logistic | 1e-4, 0.1, 500       | 50k, 80/20               | 42   | 0.224      | 0.25      | 0.454           | –          | first working pipeline (single split: optimistic, see #2)     |
| 2  | 26.10 | reg_logistic | 1e-4, 0.1, 500       | 50k, 5-fold CV           | 42   | 0.223      | 0.22      | 0.430 ± 0.005   | –          | vs #1: same config with CV → reference score                  |
| 3  | 26.10 | reg_logistic | 1e-4, 0.1, 500       | 50k, 5-fold CV (folds B) | 42   | 0.223      | 0.22      | 0.431 ± 0.009   | –          | iters grid, run 1                                             |
| 4  | 26.10 | reg_logistic | 1e-4, 0.1, 1000      | 50k, 5-fold CV (folds B) | 42   | 0.220      | 0.21      | 0.429 ± 0.011   | –          | vs #3: more iters → lower loss, same F1                       |
| 5  | 26.10 | reg_logistic | 1e-4, 0.1, 2000      | 50k, 5-fold CV (folds B) | 42   | 0.219      | 0.21      | 0.430 ± 0.011   | –          | vs #3: same F1                                                |
| 6  | 26.10 | reg_logistic | 1e-4, 0.1, 500       | 50k, 5-fold CV (folds C) | 42   | 0.223      | 0.20      | 0.429 ± 0.013   | –          | iters grid, run 2 (different folds)                           |
| 7  | 26.10 | reg_logistic | 1e-4, 0.1, 1000      | 50k, 5-fold CV (folds C) | 42   | 0.220      | 0.20      | 0.432 ± 0.012   | –          | vs #6: +0.003, within noise                                   |
| 8  | 26.10 | reg_logistic | 1e-4, 0.1, 2000      | 50k, 5-fold CV (folds C) | 42   | 0.219      | 0.21      | 0.431 ± 0.013   | –          | vs #6: within noise → 500 iters is enough                     |
| 9  | 27.10 | reg_logistic | 1e-4, 0.1, 500       | 100k, 5-fold CV (folds D) | 42   | 0.222      | 0.21      | 0.429 ± 0.012   | –          | iters grid, run 3 (100k subsample) |
| 10 | 27.10 | reg_logistic | 1e-4, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | 0.219      | 0.20      | 0.429 ± 0.012   | –          | vs #9: same F1 |
| 11 | 27.10 | reg_logistic | 1e-4, 0.1, 2000      | 100k, 5-fold CV (folds D) | 42   | 0.219      | 0.19      | 0.430 ± 0.011   | –          | vs #9: +0.001, noise → confirms 500–1000 iters enough |
| 12 | 27.10 | reg_logistic | 1e-4, 0.01, 1000     | 100k, 5-fold CV (folds D) | 42   | 0.266      | 0.28      | 0.423 ± 0.013   | –          | γ grid: γ too small, not converged (high loss) |
| 13 | 27.10 | reg_logistic | 1e-4, 0.05, 1000     | 100k, 5-fold CV (folds D) | 42   | 0.222      | 0.21      | 0.429 ± 0.012   | –          | vs #12: converged; = #9 (same γ·iters). γ grid interrupted after this (0.1/0.5/1.0 not run) |
| 14 | 27.10 | reg_logistic | 0, 0.1, 1000         | sub, 5-fold CV (folds E) | 42   | 0.220      | 0.21      | 0.430 ± 0.009   | –          | λ grid; λ=0 = plain logistic |
| 15 | 27.10 | reg_logistic | 1e-6, 0.1, 1000      | sub, 5-fold CV (folds E) | 42   | 0.220      | 0.21      | 0.430 ± 0.009   | –          | vs #14: identical |
| 16 | 27.10 | reg_logistic | 1e-5, 0.1, 1000      | sub, 5-fold CV (folds E) | 42   | 0.220      | 0.21      | 0.430 ± 0.009   | –          | vs #14: identical (picked as best by grid) |
| 17 | 27.10 | reg_logistic | 1e-4, 0.1, 1000      | sub, 5-fold CV (folds E) | 42   | 0.220      | 0.21      | 0.430 ± 0.008   | –          | vs #14: identical |
| 18 | 27.10 | reg_logistic | 1e-3, 0.1, 1000      | sub, 5-fold CV (folds E) | 42   | 0.221      | 0.21      | 0.429 ± 0.007   | –          | vs #14: −0.001, noise |
| 19 | 27.10 | reg_logistic | 1e-2, 0.1, 1000      | sub, 5-fold CV (folds E) | 42   | 0.241      | 0.25      | 0.429 ± 0.006   | –          | vs #14: same F1, threshold shifts up |
| 20 | 27.10 | reg_logistic | 1e-1, 0.1, 1000      | sub, 5-fold CV (folds E) | 42   | 0.376      | 0.40      | 0.423 ± 0.007   | –          | vs #14: −0.007, too much regularization. → λ ≤ 1e-2 doesn't matter |
| 21 | 27.10 | reg_logistic | 0, 0.01, 500         | sub, 20-fold CV (folds G) | 42   | 0.328      | 0.36      | 0.429 ± 0.015   | –          | full grid, 20 folds; not converged |
| 22 | 27.10 | reg_logistic | 0, 0.01, 1000        | sub, 20-fold CV (folds G) | 42   | 0.267      | 0.29      | 0.432 ± 0.014   | –          | vs #21: more iters |
| 23 | 27.10 | reg_logistic | 0, 0.01, 2000        | sub, 20-fold CV (folds G) | 42   | 0.237      | 0.24      | 0.433 ± 0.015   | –          | vs #21 |
| 24 | 27.10 | reg_logistic | 0, 0.05, 500         | sub, 20-fold CV (folds G) | 42   | 0.232      | 0.24      | 0.433 ± 0.015   | –          | vs #21: larger γ |
| 25 | 27.10 | reg_logistic | 0, 0.05, 1000        | sub, 20-fold CV (folds G) | 42   | 0.223      | 0.22      | 0.436 ± 0.016   | –          | best of 20-fold grid (tie) |
| 26 | 27.10 | reg_logistic | 0, 0.05, 2000        | sub, 20-fold CV (folds G) | 42   | 0.220      | 0.20      | 0.436 ± 0.017   | –          | = #25 |
| 27 | 27.10 | reg_logistic | 0, 0.1, 500          | sub, 20-fold CV (folds G) | 42   | 0.223      | 0.22      | 0.436 ± 0.016   | –          | = #25 (same γ·iters) |
| 28 | 27.10 | reg_logistic | 0, 0.1, 1000         | sub, 20-fold CV (folds G) | 42   | 0.220      | 0.20      | 0.436 ± 0.017   | –          | = #25. Grid interrupted here; 0.436 vs 0.430 (5-fold) is within noise, different folds |
| 29 | 27.10 | mse_gd       | –, 0.01, 1000        | sub, 5-fold CV (folds E) | 42   | 0.0334     | 0.22      | 0.421 ± 0.005   | –          | model comparison (single runs) |
| 30 | 27.10 | mse_sgd      | –, 0.001, 1000       | sub, 5-fold CV (folds E) | 42   | 0.0417     | 0.20      | 0.320 ± 0.032   | –          | unstable, much worse |
| 31 | 27.10 | least_squares | –, –, –              | sub, 5-fold CV (folds E) | 42   | 0.0333     | 0.23      | 0.422 ± 0.004   | –          | closed form |
| 32 | 27.10 | ridge        | 1e-4, –, –           | sub, 5-fold CV (folds E) | 42   | 0.0333     | 0.23      | 0.422 ± 0.004   | –          | = #31 |
| 33 | 27.10 | logistic     | –, 0.1, 1000         | sub, 5-fold CV (folds E) | 42   | 0.220      | 0.20      | 0.430 ± 0.009   | –          | = #14 |
| 34 | 27.10 | reg_logistic | 1e-4, 0.1, 1000      | sub, 5-fold CV (folds E) | 42   | 0.220      | 0.21      | 0.430 ± 0.008   | –          | = #17 (rerun, reproducible) |
| 35 | 27.10 | least_squares | –, –, –              | 100k, 5-fold CV (folds D) | 42   | 0.0330     | 0.23      | 0.422 ± 0.010   | –          | compare_configs, all models on same folds |
| 36 | 27.10 | ridge        | 1e-4, –, –           | 100k, 5-fold CV (folds D) | 42   | 0.0330     | 0.23      | 0.422 ± 0.010   | –          | = #35 |
| 37 | 27.10 | ridge        | 1e-2, –, –           | 100k, 5-fold CV (folds D) | 42   | 0.0330     | 0.21      | 0.422 ± 0.010   | –          | = #35 |
| 38 | 27.10 | ridge        | 1e-1, –, –           | 100k, 5-fold CV (folds D) | 42   | 0.0333     | 0.20      | 0.421 ± 0.011   | –          | vs #35: −0.001 → λ irrelevant for ridge |
| 39 | 27.10 | mse_gd       | –, 0.001, 1000       | 100k, 5-fold CV (folds D) | 42   | 0.0339     | 0.18      | 0.418 ± 0.012   | –          | not converged |
| 40 | 27.10 | mse_gd       | –, 0.01, 1000        | 100k, 5-fold CV (folds D) | 42   | 0.0331     | 0.22      | 0.421 ± 0.010   | –          | vs #39 |
| 41 | 27.10 | mse_gd       | –, 0.05, 1000        | 100k, 5-fold CV (folds D) | 42   | 0.0330     | 0.22      | 0.423 ± 0.010   | –          | ≈ least_squares (#35), converged |
| 42 | 27.10 | mse_sgd      | –, 1e-4, 1000        | 100k, 5-fold CV (folds D) | 42   | 0.0388     | 0.09      | 0.373 ± 0.012   | –          | too few steps (1 sample / step) |
| 43 | 27.10 | mse_sgd      | –, 1e-3, 1000        | 100k, 5-fold CV (folds D) | 42   | 0.0418     | 0.18      | 0.286 ± 0.046   | –          | noisy, worse |
| 44 | 27.10 | mse_sgd      | –, 1e-2, 1000        | 100k, 5-fold CV (folds D) | 42   | diverged   | –         | 0.162 ± 0.001   | –          | DIVERGED (loss ~1e45) → predicts all positive |
| 45 | 27.10 | mse_sgd      | –, 1e-3, 10000       | 100k, 5-fold CV (folds D) | 42   | 12.7       | 0.34      | 0.302 ± 0.051   | –          | fold 3 blew up (loss 63) → SGD unusable without decay |
| 46 | 27.10 | logistic     | –, 0.05, 1000        | 100k, 5-fold CV (folds D) | 42   | 0.222      | 0.21      | 0.429 ± 0.012   | –          |  |
| 47 | 27.10 | logistic     | –, 0.1, 1000         | 100k, 5-fold CV (folds D) | 42   | 0.219      | 0.20      | 0.429 ± 0.012   | –          | best by compare_configs (tie with #46, #48) |
| 48 | 27.10 | logistic     | –, 0.5, 1000         | 100k, 5-fold CV (folds D) | 42   | 0.218      | 0.19      | 0.429 ± 0.011   | –          | γ=0.5 still stable |
| 49 | 27.10 | reg_logistic | 1e-4, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | 0.219      | 0.21      | 0.429 ± 0.012   | –          | = #10 (same folds) → logistic family best by ~+0.007 over linear |
| 50 | 27.10 | reg_logistic | 1e-4, 0.1, 1000      | full (328k), 5-fold CV   | 42   | 0.221      | 0.20      | 0.429 ± 0.002   | –          | vs #49: same F1 on full data, std much lower → subsample estimate is reliable |
| 51 |       |              |                      |                          |      |            |           |                 |            |                                                              |

## Fold sets
- **D**: 100k subsample, 5-fold (`N_SUB = 100000`, cell output `(100000, 321)`)
- **E**: subsample, 5-fold, earlier kernel session (size not recorded in the notebook output)
- **G**: subsample, 20-fold (`K_FOLD = 20`), run interrupted after 8 configs
- Same letter = same folds, so rows are directly comparable only within a letter.

## Findings so far (27.10)
- **Iterations**: 500–2000 all give the same F1 once γ·iters ≳ 50 (#9–11, #23–28).
- **γ**: 0.01 is too slow at 1000 iters (#12); 0.05–0.5 all converge to the same F1.
- **λ**: no effect up to 1e-2; 1e-1 hurts (−0.007, #20) → regularization not needed at this feature count.
- **Models**: logistic ≈ reg_logistic (0.429–0.430) > least_squares ≈ ridge ≈ mse_gd (0.421–0.423) ≫ mse_sgd (≤ 0.37, unstable/diverges at γ=1e-2).
- **Full data** (#50): 0.429 ± 0.002 → the subsample estimate holds, and the threshold stays at ~0.20.

## Next steps (plan, 28.10)

**Keep a change only if it gains more than ~0.01**, or if it wins on **every fold** (see "How to compare" below).

#### ✅ Done (log #1–50)

| Step | Result |
|---|---|
| iters / γ / λ tuning | no effect once converged (γ·iters ≳ 50); λ irrelevant up to 1e-2, 1e-1 hurts |
| Model comparison | logistic ≈ reg_logistic **0.429–0.430** > least_squares ≈ ridge ≈ mse_gd 0.421–0.423 ≫ mse_sgd (≤ 0.37, unstable) |
| Full data (328k) | reg_logistic 0.429 ± 0.002, threshold ≈ 0.20 → the subsample estimate is reliable |

**Conclusion: the model is chosen; stop tuning it. The next gains come from the features.**

---

#### ⬜ Next: fixed model, change the features

**Fixed baseline for every test:** `logistic`, γ = 0.5, max_iters = 500 (stable in #48, ~2× faster),
100k subsample, 5-fold, folds D, run with `compare_configs` → directly comparable with log #35–49.

| # | Test | Values | Why |
|---|---|---|---|
| 1 | `max_missing` | 0.5, **0.8** (current), 0.95, 1.0 | cheap, potentially big: decides how many of the 321 columns survive |
| 2 | "was missing" flags | on / off | not answering a question can carry information |
| 3 | polynomial features (x², x³) on **numerical** columns only, standardized with the training-fold stats | degree 1, 2, 3 | most promising: a linear model can't learn curved effects (risk rising faster at older ages) |
| 3b | re-tune λ with the best degree | 0, 1e-4, 1e-3, 1e-2 | λ did nothing so far, but with many more columns regularization can start to matter |
| 4 | weight the positives in the logistic gradient | ×1, 3, 5, 10 | small gain expected, the threshold tuning already compensates the 9% positives |

Then: interactions (age × general health, age × sex) and log transforms (minutes of activity, drinks per week),
remove duplicated features (age, BMI, race, physical activity…), and finally confirm the best 2–3 configs on the
full data, update `run.py` (model, parameters, threshold) and submit to AIcrowd.

**Skip:** more γ / iters grids, mse_sgd, ridge vs least_squares — already answered.

#### Practical notes
- **Code change needed for tests 1–3:** `prepare_features` (`cross_validation.py`) calls `fit_preprocessing(x_tr, names)`
  without options, so `max_missing` / flags / degree can't be varied from the notebook yet.
  Pass a `prep` dict through each config, e.g. `("logistic", {"gamma": 0.5, "max_iters": 500}, {"max_missing": 0.95})`.
- **How to compare:** the ± 0.01 std is mostly fold-to-fold difficulty. On identical folds, compare the **per-fold
  differences**: a config that wins on all 5 folds by +0.004 is a real gain, even though it is below the 0.01 rule.

## Column guide
- **Model**: least_squares, ridge, logistic, reg_logistic, mse_gd, mse_sgd
- **Params**: λ, γ, iters; write "–" for what the model doesn't use (e.g. "1e-3, –, –" for ridge)
- **Setup**: data size + validation method (e.g. "50k, 80/20" or "full, 5-fold CV")
- **Train loss**: mean over the folds with CV
- **F1 (mean ± std)**: just the value for 80/20; mean ± std with CV (e.g. "0.452 ± 0.008")
- **AIcrowd F1**: only for submitted runs (team limit: 5 per day)
- **Notes**: what you changed vs the previous run (including any preprocessing change), and what you concluded

## Rules
1. Change **one thing** at a time compared with a previous row.
2. Never delete a row, even a bad result: bad results are also findings.
3. Note the **row number** you compare against in Notes (e.g. "vs #1: γ 0.5 → same F1, faster").
