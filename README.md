# longest_run

© 2025 Paolo Bonfini — All content in this repository is original work. Forked from: https://github.com/gitpeblo/longest_run

This repository provides a Python implementation of the "Longest Run" (LR) metric, designed for evaluating the robustness and predictive reach of forecasting models on time series data. The metric estimates the number of timesteps for which a model remains statistically consistent with the ground truth before a significant divergence occurs.

## Overview

The LR metric identifies the last timestep at which a model's predictions are still statistically consistent with the true data. This is achieved by progressively sliding a window along the time axis and applying statistical tests to assess model validity.

The final LR score is computed as:

```
LR = base_score_LR / t_LR
```

Where:
- `base_score_LR` is the selected error metric (e.g. MAE) up to the point of divergence.
- `t_LR` is the timestep at which the model first significantly diverges from the data.

## Key Features

- Multiple statistical tests for detecting divergence:
  - Chi-squared (chi2, chi2nu)
  - t-tests
  - Kolmogorov-Smirnov (KS)
  - Pearson correlation
- Optional smoothing of signals
- Automatic window estimation using wavelet analysis
- Handles both precomputed predictions and recursive model predictions
- Visualization support (including interactive widgets in Jupyter)

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- prettytable
- ipywidgets
- IPython (for display)
- Internal modules under `utils/`

## Usage

### Option 1: Using Precomputed Predictions

```python
result = longest_run(
    y_roll=y_true,
    yhat_roll=y_pred,
    method='KS',
    alpha=0.01,
    plot='results'
)
```

### Option 2: Using a Model

```python
result = longest_run(
    model=my_model,
    x_0=initial_window,
    y=target_matrix,
    method='chi2nu',
    base_metric='MAE',
    smooth_algo='LOWESS',
    plot='all_widget'
)
```

### Important Parameters

| Parameter         | Description |
|-------------------|-------------|
| `model`           | Model object with a `.predict()` method |
| `x_0`             | Initial feature window for recursive prediction |
| `y`               | True target data matrix |
| `y_roll`          | Manually supplied test target data |
| `yhat_roll`       | Manually supplied predicted values |
| `method`          | Statistical test: `'chi2'`, `'chi2nu'`, `'ttest'`, `'KS'`, `'rho'` |
| `alpha`           | Significance threshold for p-value based tests |
| `rho_thresh`      | Correlation threshold for `'rho'` method |
| `base_metric`     | Base metric for LR score: `'MAE'`, `'SMAPE'`, `'MSE'`, `'RMSE'`, `'r2'` |
| `stride`          | Time stride between windows in `y` |
| `smooth_algo`     | Smoothing method: `'LOWESS'` or `'SG'` |
| `kernel_size`     | Smoothing window size (optional, auto-estimated if None) |
| `burn_in`         | Minimum timesteps before applying tests |
| `MAX_timesteps`   | Max number of forecast timesteps |
| `plot`            | Plot mode: `'None'`, `'results'`, `'results_widget'`, `'all'`, `'all_widget'` |
| `verbose`         | Verbosity level: `0`, `1`, or `2` |
| `debug`           | If True, print all statistics per timestep |

## Returns

The function returns a dictionary with the following structure:

| Key                   | Description |
|------------------------|-------------|
| `kernel_size`         | Smoothing kernel size used |
| `win_local_proposed`  | Initial suggested window size from wavelet analysis |
| `win_local`           | Final adopted local window size |
| `df_stats`            | DataFrame with per-timestep p-values, statistics, and base scores |
| `stat_LR`             | Test statistic value at the LR point |
| `p_LR`                | P-value at the LR point |
| `t_LR`                | Timestep where divergence is first detected |
| `base_score_LR`       | Base metric value up to `t_LR` |
| `LR`                  | Final LR score: base score divided by `t_LR` |
| `base_metric`         | Name of the base metric used |
| `df_ysm`              | Smoothed target signal |
| `df_ysm_std`          | Estimated standard deviation of the smoothed signal |

This structure allows you to inspect the statistical trajectory, visualize divergence, and compare multiple models or configurations.

## Example

Minimal example with model-based recursive prediction:

```python
from longest_run import longest_run

result = longest_run(
    model=trained_model,
    x_0=initial_data,
    y=sliding_targets,
    method='chi2nu',
    plot='results'
)

print(result['LR'], result['t_LR'])
```

