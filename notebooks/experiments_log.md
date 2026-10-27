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
| 9  |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 10 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 11 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 12 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 13 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 14 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 15 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 16 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 17 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 18 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 19 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 20 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 21 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 22 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 23 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 24 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 25 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 26 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 27 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 28 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 29 |       |              |                      |                          |      |            |           |                 |            |                                                              |
| 30 |       |              |                      |                          |      |            |           |                 |            |                                                              |

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
