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
| 51 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.21      | 0.428 ± 0.013   | –          | **no expansion** = reference for #52–80. First row with the threshold chosen on the training folds: vs #49 −0.001 |
| 52 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.21      | 0.429 ± 0.012   | –          | **inter 2 top 10** (+45 products): vs #51 +0.001, noise |
| 53 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.426 ± 0.010   | –          | **inter 2 top 20** (+190): vs #51 −0.002 |
| 54 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.425 ± 0.010   | –          | **inter 2 top 30** (+435): vs #51 −0.003 → too many products overfit |
| 55 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.21      | 0.429 ± 0.010   | –          | **poly 2 all** (+84 squares): vs #51 +0.001, noise |
| 56 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.428 ± 0.012   | –          | **poly 2 all + inter 2 top 10**: = #51 |
| 57 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.429 ± 0.009   | –          | **poly 2 all + inter 2 top 20** (+84 +190): vs #51 +0.001, lowest std of the 0.429 rows → config of run.py (#81) |
| 58 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.425 ± 0.010   | –          | **poly 2 all + inter 2 top 30**: vs #51 −0.003 |
| 59 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.21      | 0.429 ± 0.012   | –          | **poly 2 top 10** (+10 squares): vs #51 +0.001, noise |
| 60 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.428 ± 0.012   | –          | **poly 2 top 10 + inter 2 top 10**: = #51 |
| 61 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.428 ± 0.010   | –          | **poly 2 top 10 + inter 2 top 20**: = #51 |
| 62 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.425 ± 0.010   | –          | **poly 2 top 10 + inter 2 top 30**: vs #51 −0.003 |
| 63 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.21      | 0.428 ± 0.011   | –          | **poly 3 all** (x² and x³): = #51 → x³ adds nothing over x² (#55) |
| 64 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.427 ± 0.010   | –          | **poly 3 all + inter 2 top 10**: vs #51 −0.001 |
| 65 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.428 ± 0.009   | –          | **poly 3 all + inter 2 top 20**: = #51 |
| 66 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.23      | 0.423 ± 0.010   | –          | **poly 3 all + inter 2 top 30**: vs #51 −0.005 |
| 67 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.428 ± 0.012   | –          | **poly 3 top 10**: = #51 |
| 68 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.428 ± 0.013   | –          | **poly 3 top 10 + inter 2 top 10**: = #51 |
| 69 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.427 ± 0.010   | –          | **poly 3 top 10 + inter 2 top 20**: vs #51 −0.001 |
| 70 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.425 ± 0.010   | –          | **poly 3 top 10 + inter 2 top 30**: vs #51 −0.003 |
| 71 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.428 ± 0.013   | –          | **inter 3 top 10** (+45 pairs +120 triples): = #51 |
| 72 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.425 ± 0.012   | –          | **inter 3 top 15** (+105 pairs +455 triples): vs #51 −0.003 |
| 73 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.23      | 0.428 ± 0.011   | –          | **poly 2 all + inter 3 top 10**: = #51 |
| 74 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.424 ± 0.008   | –          | **poly 2 all + inter 3 top 15**: vs #51 −0.004 |
| 75 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.429 ± 0.013   | –          | **poly 2 top 10 + inter 3 top 10**: vs #51 +0.001, noise |
| 76 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.425 ± 0.009   | –          | **poly 2 top 10 + inter 3 top 15**: vs #51 −0.003 |
| 77 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.426 ± 0.011   | –          | **poly 3 all + inter 3 top 10**: vs #51 −0.002 |
| 78 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.23      | 0.423 ± 0.008   | –          | **poly 3 all + inter 3 top 15**: vs #51 −0.005 |
| 79 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.22      | 0.429 ± 0.014   | –          | **poly 3 top 10 + inter 3 top 10**: vs #51 +0.001, noise (highest std) |
| 80 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 100k, 5-fold CV (folds D) | 42   | –          | 0.23      | 0.424 ± 0.010   | –          | **poly 3 top 10 + inter 3 top 15**: vs #51 −0.004. End of the expansion grid |
| 81 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | full (328k), run.py (no validation) | 42   | 0.220      | 0.22      | –               | *to fill*  | **poly 2 all + inter 2 top 20** = config of #57 (349 → 624 columns). Threshold chosen on the training rows. 14,259 / 109,379 test rows predicted positive (13.0%); 2.2% of the predictions differ from run.py without expansion. `submissions/submission_poly2_inter20.csv` |
| 82 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.23      | 0.430 ± 0.023   | –          | **no expansion** = reference for #83–102. 20k rows: std ~0.02, twice the 100k one. Same F1 in compare_configs |
| 83 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.429 ± 0.020   | –          | **inter 2 top 10**: vs #82 −0.001, noise. Same F1 in compare_configs |
| 84 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.25      | 0.426 ± 0.016   | –          | **inter 2 top 20**: vs #82 −0.004 |
| 85 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.419 ± 0.016   | –          | **inter 2 top 30**: vs #82 −0.011 (only −0.003 at 100k, #54) |
| 86 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.425 ± 0.023   | –          | **poly 2 all**: vs #82 −0.005. Same F1 in compare_configs |
| 87 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | 0.215      | 0.25      | 0.422 ± 0.020   | –          | **poly 2 all + inter 2 top 10**: vs #82 −0.008. Per fold 0.395 / 0.406 / 0.450 / 0.427 / 0.433; the oracle threshold would give 0.434 (+0.012). Same F1 in the single-config cell and in compare_configs |
| 88 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.26      | 0.421 ± 0.015   | –          | **poly 2 all + inter 2 top 20** (config of run.py, #81): vs #82 −0.009 (+0.001 at 100k, #57) |
| 89 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.25      | 0.415 ± 0.018   | –          | **poly 2 all + inter 2 top 30**: vs #82 −0.015 |
| 90 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.23      | 0.429 ± 0.024   | –          | **poly 2 top 10**: vs #82 −0.001, noise |
| 91 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.429 ± 0.023   | –          | **poly 2 top 10 + inter 2 top 10**: vs #82 −0.001, noise |
| 92 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.428 ± 0.018   | –          | **poly 2 top 10 + inter 2 top 20**: vs #82 −0.002 |
| 93 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.26      | 0.415 ± 0.015   | –          | **poly 2 top 10 + inter 2 top 30**: vs #82 −0.015 |
| 94 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.426 ± 0.020   | –          | **poly 3 all**: vs #82 −0.004 |
| 95 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.25      | 0.422 ± 0.021   | –          | **poly 3 all + inter 2 top 10**: vs #82 −0.008 |
| 96 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.27      | 0.415 ± 0.014   | –          | **poly 3 all + inter 2 top 20**: vs #82 −0.015 |
| 97 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.26      | 0.412 ± 0.023   | –          | **poly 3 all + inter 2 top 30**: vs #82 −0.018 |
| 98 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.23      | 0.432 ± 0.024   | –          | **poly 3 top 10**: vs #82 +0.002, best of the 20k grid but noise (= no expansion at 100k, #67). Same F1 in compare_configs |
| 99 | 28.10 | reg_logistic | 1e-3, 0.1, 1000      | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.426 ± 0.021   | –          | **poly 3 top 10 + inter 2 top 10**: vs #82 −0.004 |
| 100 | 28.10 | reg_logistic | 1e-3, 0.1, 1000     | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.429 ± 0.017   | –          | **poly 3 top 10 + inter 2 top 20**: vs #82 −0.001, noise |
| 101 | 28.10 | reg_logistic | 1e-3, 0.1, 1000     | 20k, 5-fold CV (folds H)  | 42   | –          | 0.27      | 0.409 ± 0.021   | –          | **poly 3 top 10 + inter 2 top 30**: vs #82 −0.021, worst of the 20k grid |
| 102 | 28.10 | ridge        | 1e-4, –, –          | 20k, 5-fold CV (folds H)  | 42   | –          | 0.24      | 0.420 ± 0.016   | –          | **poly 2 all + inter 2 top 10** (compare_configs): vs #87 −0.002, noise |
| 103 |       |              |                      |                          |      |            |           |                 |            |                                                              |

## Fold sets
- **D**: 100k subsample, 5-fold (`N_SUB = 100000`, cell output `(100000, 321)`)
- **E**: subsample, 5-fold, earlier kernel session (size not recorded in the notebook output)
- **G**: subsample, 20-fold (`K_FOLD = 20`), run interrupted after 8 configs
- **H**: 20k subsample, 5-fold (`N_SUB = 20000`, cell output `(20000, 321)`, folds of 3999–4001 rows)
- Same letter = same folds, so rows are directly comparable only within a letter.
- **Threshold rule (from #51):** `cross_validate` now picks the threshold on the training folds and applies it to the
  validation fold, like `run.py` does for the test set. #1–50 picked it on the validation fold itself, which is slightly
  optimistic (about −0.001 on the same folds: #51 vs #49). The gap grows on small data: on 20k rows the
  validation-fold threshold would give +0.012 (#87).

## Findings so far (28.10)
- **Iterations**: 500–2000 all give the same F1 once γ·iters ≳ 50 (#9–11, #23–28).
- **γ**: 0.01 is too slow at 1000 iters (#12); 0.05–0.5 all converge to the same F1.
- **λ**: no effect up to 1e-2; 1e-1 hurts (−0.007, #20) → regularization not needed at this feature count.
- **Models**: logistic ≈ reg_logistic (0.429–0.430) > least_squares ≈ ridge ≈ mse_gd (0.421–0.423) ≫ mse_sgd (≤ 0.37, unstable/diverges at γ=1e-2).
- **Full data** (#50): 0.429 ± 0.002 → the subsample estimate holds, and the threshold stays at ~0.20.
- **Feature expansion** (#51–80, folds D, reg_logistic λ = 1e-3): **no gain beyond noise**. The best rows reach 0.429
  (#52, #55, #57, #59, #75, #79) vs 0.428 without expansion (#51), with a std of 0.009–0.014. x³ never beats x².
  Large product sets overfit: pairs of the top 30 or triples of the top 15 lose 0.003–0.005 in every combination.
- **Same grid on 20k rows** (#82–101, folds H, std ~0.02): same picture, nothing beats no expansion (0.430) beyond
  noise, but the large sets lose much more (−0.011 to −0.021, vs −0.003 to −0.005 at 100k) → the overfitting shrinks
  with more rows, so judge an expansion on ≥ 100k rows, not on 20k.
- **Full data with x² + pairs of the top 20** (#81, config of #57): threshold 0.22, 13.0% of the test rows predicted
  positive, 2.2% of the predictions changed vs no expansion. AIcrowd score: to fill.

## Next steps (plan, 28.10)

**Keep a change only if it gains more than ~0.01**, or if it wins on **every fold** (see "How to compare" below).

#### ✅ Done (log #1–102)

| Step | Result |
|---|---|
| iters / γ / λ tuning | no effect once converged (γ·iters ≳ 50); λ irrelevant up to 1e-2, 1e-1 hurts |
| Model comparison | logistic ≈ reg_logistic **0.429–0.430** > least_squares ≈ ridge ≈ mse_gd 0.421–0.423 ≫ mse_sgd (≤ 0.37, unstable) |
| Full data (328k) | reg_logistic 0.429 ± 0.002, threshold ≈ 0.20 → the subsample estimate is reliable |
| Polynomial + interactions (#51–80 on 100k, #82–102 on 20k) | no gain: best 0.429 vs 0.428 without expansion (100k); large product sets overfit (−0.003 to −0.005 at 100k, up to −0.021 at 20k). x² + pairs of the top 20 run on the full data with run.py (#81) |

**Conclusion: the model is chosen; stop tuning it. The next gains come from the features.**

---

#### ⬜ Next: fixed model, change the features

**Fixed baseline for every test:** `logistic`, γ = 0.5, max_iters = 500 (stable in #48, ~2× faster),
100k subsample, 5-fold, folds D, run with `compare_configs` → directly comparable with log #35–49.

| # | Test | Values | Why |
|---|---|---|---|
| 1 | `max_missing` | 0.5, **0.8** (current), 0.95, 1.0 | cheap, potentially big: decides how many of the 321 columns survive |
| 2 | "was missing" flags | on / off | not answering a question can carry information |
| 3 | ✅ **done (#51–80, #82–101): no gain**. Polynomial features (x², x³) on **numerical** columns only, standardized with the training-fold stats | degree 1, 2, 3 | most promising: a linear model can't learn curved effects (risk rising faster at older ages) |
| 3b | re-tune λ with the best degree | 0, 1e-4, 1e-3, 1e-2 | λ did nothing so far, but with many more columns regularization can start to matter |
| 4 | weight the positives in the logistic gradient | ×1, 3, 5, 10 | small gain expected, the threshold tuning already compensates the 9% positives |

Then: interactions (products of the strongest features: done in #51–101, no gain; hand-picked ones such as
age × general health or age × sex not tried yet) and log transforms (minutes of activity, drinks per week),
remove duplicated features (age, BMI, race, physical activity…), and finally confirm the best 2–3 configs on the
full data, update `run.py` (model, parameters, threshold) and submit to AIcrowd.

**Skip:** more γ / iters grids, mse_sgd, ridge vs least_squares — already answered.

#### Practical notes
- **Polynomial features (test 3) and interactions: available**, as keys of `params` (see "Feature expansion" at the end of `experiments.ipynb`).
  **Still to code for tests 1–2:** `prepare_features` calls `fit_preprocessing(x_tr, names)` without options, so
  `max_missing` / flags can't be varied yet: add them to `params` the same way.
- **How to compare:** the ± 0.01 std is mostly fold-to-fold difficulty. On identical folds, compare the **per-fold
  differences**: a config that wins on all 5 folds by +0.004 is a real gain, even though it is below the 0.01 rule.

## Column guide
- **Model**: least_squares, ridge, logistic, reg_logistic, mse_gd, mse_sgd
- **Params**: λ, γ, iters; write "–" for what the model doesn't use (e.g. "1e-3, –, –" for ridge)
- **Setup**: data size + validation method (e.g. "50k, 80/20" or "full, 5-fold CV")
- **Train loss**: mean over the folds with CV ("–" when the run doesn't print it, e.g. `grid_search_expansion`)
- **F1 (mean ± std)**: just the value for 80/20; mean ± std with CV (e.g. "0.452 ± 0.008")
- **AIcrowd F1**: only for submitted runs (team limit: 5 per day)
- **Notes**: what you changed vs the previous run (including any preprocessing change), and what you concluded
- **Expansion notation** (Notes, from #51): `poly d all` / `poly d top k` = powers 2..d of all / of the k strongest
  numerical columns; `inter d top k` = products of 2..d features among the k strongest. These are the params keys
  `poly_degree`, `poly_top`, `inter_degree`, `inter_top` (see "Feature expansion" in `experiments.ipynb`).

## Rules
1. Change **one thing** at a time compared with a previous row.
2. Never delete a row, even a bad result: bad results are also findings.
3. Note the **row number** you compare against in Notes (e.g. "vs #1: γ 0.5 → same F1, faster").
