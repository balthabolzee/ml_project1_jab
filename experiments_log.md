# Experiment log

| #  | Date  | Model        | Params (λ, γ, iters) | Setup (data, validation) | Seed | Train loss | Threshold | F1 (mean ± std) | AIcrowd F1 | Notes                  |
|----|-------|--------------|----------------------|--------------------------|------|------------|-----------|-----------------|------------|------------------------|
| 1  | 26.10 | reg_logistic | 1e-4, 0.1, 500       | 50k, 80/20               | 42   | 0.224      | 0.25      | 0.454           | –          | first working pipeline |
| 2  |       |              |                      |                          |      |            |           |                 |            |                        |
| 3  |       |              |                      |                          |      |            |           |                 |            |                        |
| 4  |       |              |                      |                          |      |            |           |                 |            |                        |
| 5  |       |              |                      |                          |      |            |           |                 |            |                        |
| 6  |       |              |                      |                          |      |            |           |                 |            |                        |
| 7  |       |              |                      |                          |      |            |           |                 |            |                        |
| 8  |       |              |                      |                          |      |            |           |                 |            |                        |
| 9  |       |              |                      |                          |      |            |           |                 |            |                        |
| 10 |       |              |                      |                          |      |            |           |                 |            |                        |
| 11 |       |              |                      |                          |      |            |           |                 |            |                        |
| 12 |       |              |                      |                          |      |            |           |                 |            |                        |
| 13 |       |              |                      |                          |      |            |           |                 |            |                        |
| 14 |       |              |                      |                          |      |            |           |                 |            |                        |
| 15 |       |              |                      |                          |      |            |           |                 |            |                        |
| 16 |       |              |                      |                          |      |            |           |                 |            |                        |
| 17 |       |              |                      |                          |      |            |           |                 |            |                        |
| 18 |       |              |                      |                          |      |            |           |                 |            |                        |
| 19 |       |              |                      |                          |      |            |           |                 |            |                        |
| 20 |       |              |                      |                          |      |            |           |                 |            |                        |
| 21 |       |              |                      |                          |      |            |           |                 |            |                        |
| 22 |       |              |                      |                          |      |            |           |                 |            |                        |
| 23 |       |              |                      |                          |      |            |           |                 |            |                        |
| 24 |       |              |                      |                          |      |            |           |                 |            |                        |
| 25 |       |              |                      |                          |      |            |           |                 |            |                        |
| 26 |       |              |                      |                          |      |            |           |                 |            |                        |
| 27 |       |              |                      |                          |      |            |           |                 |            |                        |
| 28 |       |              |                      |                          |      |            |           |                 |            |                        |
| 29 |       |              |                      |                          |      |            |           |                 |            |                        |
| 30 |       |              |                      |                          |      |            |           |                 |            |                        |

## Column guide
- **Model**: least_squares, ridge, logistic, reg_logistic, mse_gd, mse_sgd
- **Params**: λ, γ, iters; write "–" for what the model doesn't use (e.g. "1e-3, –, –" for ridge)
- **Setup**: data size + validation method (e.g. "50k, 80/20" or "full, 5-fold CV")
- **F1 (mean ± std)**: just the value for 80/20; mean ± std with CV (e.g. "0.452 ± 0.008")
- **AIcrowd F1**: only for submitted runs (team limit: 5 per day)
- **Notes**: what you changed vs the previous run (including any preprocessing change), and what you concluded

## Rules
1. Change **one thing** at a time compared with a previous row.
2. Never delete a row, even a bad result: bad results are also findings.
3. Note the **row number** you compare against in Notes (e.g. "vs #1: γ 0.5 → same F1, faster").
