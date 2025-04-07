# 📦 CHANGELOG — Longest Run Package

This changelog tracks key updates and version history for each module in the
`longest_run` package.

---

### v1.2.0 — 2025-04-07

## core.py

- Using double wavelet detection
- Removed obsolete 'global' evaluation type

## estimate_window.py

- Douple peak selection
- OLS instead of differencing
- Added option to export plots

---

### v1.1.0 — 2025-03-26

- Reshaped CHANGELOG.md

## core.py

- Downscaled optimal window to half the optimal wavelet scale
- Forced burn_in as hard limit of timesteps necessary to calculate statistics
- Added smoothing results to the returned results
- Added check when calcualting std when only 1 point is in win_local
- Added option to save figure

## estimate_window.py

- Downscaled optimal window to half the optimal wavelet scale

## utils/plot_results.py

- Added option to save figure

---

### v1.0.0 — 2025-03-21 *(Ported from `LRs_v5`)*

## core.py

- Initial version.

## utils/array_to_pandas.py

- Initial version.

## utils/estimate_window.py

- Initial version.

## utils/plot_results.py

- Initial version.

## utils/recursive_predict.py

- Initial version.

## utils/smooth_signal.py

- Initial version.

## utils/statistical_tests.py

- Initial version.

