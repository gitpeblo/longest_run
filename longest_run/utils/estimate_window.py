import numpy as np #type: ignore
import pandas as pd #type: ignore
import pywt #type: ignore
from statsmodels.tsa.stattools import kpss #type: ignore
from matplotlib import pyplot as plt #type: ignore
import matplotlib as mpl #type: ignore
import seaborn as sns #type: ignore
from scipy.signal import find_peaks, peak_prominences #type: ignore
from statsmodels.tsa.stattools import InterpolationWarning  #type: ignore
import warnings #type: ignore
from scipy.stats import linregress #type: ignore
from scipy.ndimage import uniform_filter1d #type: ignore

# Define the plotting style dictionary
plot_style = {
    "figure.figsize": (8, 4),
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.size": 10,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "gray",
    "axes.labelcolor": "gray",
    "xtick.color": "gray",
    "ytick.color": "gray",
    "legend.edgecolor": "gray",
    "legend.labelcolor": "gray"
}

def prepare_series_input(series):
    """
    Ensures input is a 1D numpy array, regardless of whether it's passed as a
    list, pandas Series, numpy array, or single-column DataFrame.
    """
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("DataFrame input must have exactly one column.")
        return series.iloc[:, 0].values
    elif isinstance(series, pd.Series):
        return series.values
    elif isinstance(series, np.ndarray):
        if series.ndim == 2:
            if series.shape[1] != 1:
                raise ValueError("2D numpy array input must have exactly one column.")
            return series[:, 0]
        return series
    elif isinstance(series, list):
        return np.array(series)
    else:
        raise TypeError("Unsupported input type. Must be list, numpy array, pandas Series or single-column DataFrame.")

def find_differencing_order(series, max_d=5, significance_level=0.05):
    """
    Determines the number of differences required to remove a deterministic trend 
    using the KPSS test for trend stationarity. If the test fails, the series is 
    differenced iteratively until trend stationarity is achieved or the maximum 
    differencing order is reached.

    Parameters
    ----------
    series : array-like
        The input time series data. Can be a list, numpy array, pandas Series, or 
        a single-column pandas DataFrame.
    max_d : int, optional, default=5
        The maximum number of differences to apply before stopping.
    significance_level : float, optional, default=0.005
        The p-value threshold for the KPSS test. If p-value > significance_level, 
        the series is considered trend-stationary.

    Returns
    -------
    d : int
        The number of differences applied to achieve trend stationarity.
    test_series : numpy.ndarray
        The differenced time series after applying the necessary transformations.

    Raises
    ------
    TypeError
        If the input series is not a supported data type.
    InterpolationWarning
        If the KPSS test statistic is out of range, indicating that the series is 
        already highly trend-stationary.

    Notes
    -----
    - The function uses the KPSS test with `regression="ct"` to specifically test 
      for trend stationarity.
    - If the test statistic falls outside the lookup table, it assumes the series 
      is already trend-stationary and stops differencing.
    - If KPSS fails due to numerical issues, the function exits early with the 
      last computed differenced series.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tsa.stattools import kpss
    >>> series = np.array([1, 2, 3, 5, 8, 13, 21, 34])
    >>> d, stationary_series = find_differencing_order(series)
    >>> print(d)
    1
    """

    d = 0
    test_series = series.copy()

    # Limit lags to avoid interpolation warning issues
    max_lags = min(10, len(series) // 5)
    # Rule-of-thumb: Using at most 1/5 of series length
            
    while d < max_d:
        try:
            with warnings.catch_warnings():
                # warnings.simplefilter("error", category=InterpolationWarning)  # Only turn InterpolationWarning into error
                warnings.simplefilter("ignore")  # Let other warnings pass
                _, p_value, _, _ = kpss(test_series, regression='ct', nlags=max_lags)
        except InterpolationWarning:
            return d, test_series  # Assume stationarity if interpolation fails (test unreliable)
        except Exception:
            break  # Unexpected error – break out
        
        if p_value > significance_level:
            return d, test_series

        test_series = np.diff(test_series)
        d += 1

    return d, test_series

def remove_trend_ols(series):
    """
    Removes a linear deterministic trend from the input series using 
    ordinary least squares (OLS) regression.

    This function fits a linear model of the form:
        y(t) = β₀ + β₁·t + ε(t)
    and returns the residuals ε(t), which represent the detrended series.
    This approach preserves the variance structure and avoids the distortion
    often introduced by differencing.

    Parameters
    ----------
    series : array-like
        The input time series data. Can be a list, numpy array, pandas Series, or 
        a single-column pandas DataFrame.

    Returns
    -------
    detrended_series : numpy.ndarray
        The residuals from the OLS regression, representing the signal with the 
        linear trend removed.

    Raises
    ------
    TypeError
        If the input series is not a supported data type or is not 1-dimensional.

    Notes
    -----
    - This method is less aggressive than differencing, making it more suitable
      for applications (like wavelet variance estimation) where preserving low-
      frequency structure is critical.
    - The data is not mean-centered internally, so users may wish to subtract 
      the mean afterward if required.

    Examples
    --------
    >>> import numpy as np
    >>> series = np.array([1.0, 2.1, 3.0, 4.1, 5.1])
    >>> detrended = remove_trend_ols(series)
    >>> print(detrended)
    array([-0.06..., -0.02..., -0.00...,  0.02...,  0.06...])
    """

    # Handle input formats
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("DataFrame input must have exactly one column.")
        series = series.iloc[:, 0].values
    elif isinstance(series, pd.Series):
        series = series.values
    elif isinstance(series, list):
        series = np.array(series)
    elif isinstance(series, np.ndarray):
        if series.ndim == 2:
            if series.shape[1] != 1:
                raise ValueError("2D numpy array input must have exactly one column.")
            series = series[:, 0]
    else:
        raise TypeError("Unsupported input type. Must be list, numpy array, pandas Series, or single-column DataFrame.")

    # Construct time index
    t = np.arange(len(series))

    # Fit OLS: y = β₀ + β₁·t
    slope, intercept, _, _, _ = linregress(t, series)
    trend = intercept + slope * t

    # Return residuals (detrended series)
    return series - trend

def peak_selection(mean_wsd, scales, peak_mode="main"):
    """
    Identifies a concave peak in the (smoothed) wavelet spectral density, with 
    three modes for selection, and returns both the chosen scale and the 
    smoothed spectral density array.

    1) 'main'   => Selects the most prominent concave peak.
    2) 'first'  => Selects the first (leftmost) concave peak in ascending scale.
    3) 'second' => Selects the second (next-to-leftmost) concave peak in ascending scale.

    To reduce spurious pointy peaks, the function applies a uniform (box) filter
    with size=3 on the `mean_wsd` array before detecting peaks.

    Parameters
    ----------
    mean_wsd : np.ndarray
        The original wavelet spectral density values for different scales.
    scales : np.ndarray
        Corresponding scale values (time window sizes).
    peak_mode : {'main', 'first', 'second'}, optional (default='main')
        - 'main'   => Sort peaks by descending prominence, pick the first that 
                      satisfies concavity.
        - 'first'  => Scan peaks in ascending scale order, pick the first 
                      that satisfies concavity.
        - 'second' => Scan peaks in ascending scale order, pick the second 
                      that satisfies concavity.

    Returns
    -------
    chosen_scale : float
        The chosen scale (time window size) at the identified peak.
    mean_wsd_smoothed : np.ndarray
        The spectral density array after smoothing (size=3 uniform filter).

    Raises
    ------
    ValueError
        If `scales` and `mean_wsd` have mismatched lengths.

    Notes
    -----
    1. **Smoothing**:
       - We smooth `mean_wsd` with a uniform kernel of size=3 (using 
         `uniform_filter1d`) to avoid tiny, sharp, noise-induced peaks.
    
    2. **Peak Detection**:
       - We run `find_peaks` on the **smoothed** array. The peak indices 
         still correspond to the same positions in `scales`.

    3. **Concavity Check**:
       - We compute the second derivative (also on the smoothed array) 
         via `d2S_dT2 = np.diff(smoothed_wsd, n=2)`.
       - A peak at index `pk` is "concave" if `d2S_dT2[pk - 1] < 0`.

    4. **Modes**:
       - 'first'  : Pick the earliest peak in ascending scale order that meets 
                   the concavity condition.
       - 'second' : Pick the second earliest peak (if it exists).
       - 'main'   : Pick the most prominent among all concave peaks 
                   (sorted by descending prominence).

    5. **Fallback**:
       - If no concave peak is found in 'first' or 'second' mode, fallback to the 
         absolute maximum in the **smoothed** array.
       - If no concave peak is found in 'main' mode, fallback to the 
         most prominent peak ignoring concavity.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import find_peaks, peak_prominences
    >>> mean_wsd = np.array([0.05, 1.2, 0.9, 3.5, 5.0, 4.8, 0.3, 0.2])
    >>> scales = np.arange(1, len(mean_wsd) + 1)
    >>> # 'main' mode => biggest concave peak
    >>> scale_main, smoothed = peak_selection(mean_wsd, scales, peak_mode='main')
    >>> # 'first' mode => first valid concave peak
    >>> scale_first, smoothed = peak_selection(mean_wsd, scales, peak_mode='first')
    >>> # 'second' mode => second valid concave peak
    >>> scale_second, smoothed = peak_selection(mean_wsd, scales, peak_mode='second')
    """
    if len(scales) != len(mean_wsd):
        raise ValueError("`scales` and `mean_wsd` must have the same length.")

    # 1) Smooth the spectrum with kernel size=3 to reduce spiky noise
    mean_wsd_smoothed = uniform_filter1d(mean_wsd, size=3)

    # 2) Compute second derivative on the smoothed data
    d2S_dT2 = np.diff(mean_wsd_smoothed, n=2)

    # 3) Identify local maxima on the smoothed array
    peaks, _ = find_peaks(mean_wsd_smoothed)
    if len(peaks) == 0:
        # fallback => global max in smoothed array
        chosen_index = np.argmax(mean_wsd_smoothed)
        return scales[chosen_index], mean_wsd_smoothed

    # ----------------------------------------------------------------------
    # 'first' and 'second' modes (scan in ascending order of scale)
    # ----------------------------------------------------------------------
    if peak_mode in ["first", "second"]:
        sorted_peaks = sorted(peaks)
        valid_peaks = []
        for pk in sorted_peaks:
            if 1 <= pk < (len(mean_wsd_smoothed) - 1):
                if d2S_dT2[pk - 1] < 0:  # concavity check
                    valid_peaks.append(pk)
            if len(valid_peaks) >= 2:
                break
        if peak_mode == "first" and valid_peaks:
            chosen_peak = valid_peaks[0]
        elif peak_mode == "second" and len(valid_peaks) >= 2:
            chosen_peak = valid_peaks[1]
        elif valid_peaks:
            chosen_peak = valid_peaks[0]  # fallback: first
        else:
            chosen_peak = np.argmax(mean_wsd_smoothed)  # fallback: global max
        return scales[chosen_peak], mean_wsd_smoothed

    # ----------------------------------------------------------------------
    # 'main' mode
    # ----------------------------------------------------------------------
    prominences = peak_prominences(mean_wsd_smoothed, peaks)[0]
    sorted_peaks = sorted(zip(peaks, prominences), key=lambda x: -x[1])

    chosen_peak = None
    for (pk, _) in sorted_peaks:
        if 1 <= pk < (len(mean_wsd_smoothed) - 1):
            if d2S_dT2[pk - 1] < 0:  # concavity
                chosen_peak = pk
                break

    if chosen_peak is None:
        # fallback: pick the most prominent ignoring concavity
        chosen_peak = sorted_peaks[0][0]

    return scales[chosen_peak], mean_wsd_smoothed

def estimate_window(
    series,
    peak_mode="main",
    detrend=True,
    plot_detrended=False,
    plot_spectrum=False,
    plot_label="Data",
    savefig=None,
):
    """
    Estimates the optimal rolling window size for variance stabilization using
    wavelet spectral density analysis. This function identifies the dominant
    time scale over which the variance structure remains stable.

    The method applies:
    - **Optionally** linear detrending so that the CWT focuses on variance
    - **Wavelet Transform** to extract the dominant periodicity in local variance.
    - **Peak selection** (with concavity) to identify the most
      significant or first stable scale (depending on `peak_mode`).
    - **Visualization** to interpret the wavelet power spectrum.

    Relationship Between Wavelet Power Spectrum and Window Size
    -----------------------------------------------------------
    - The **wavelet power spectrum** represents the energy distribution of the
      time series across different scales (window sizes).
    - The higher the power at a given scale, the more variance is explained by
      that window size.
    - The **optimal scale** corresponds to the **peak of the power spectrum**, 
      indicating the characteristic timescale over which fluctuations stabilize.

    Parameters
    ----------
    series : array-like
        The input time series data, which can be a list, numpy array, pandas Series, 
        or single-column DataFrame.
    peak_mode : {'main', 'first'}, optional (default='main')
        - 'main'  => Picks the **most prominent** concave peak from the wavelet
                     spectrum.
        - 'first' => Picks the **leftmost** concave peak, i.e. smallest scale
                     that satisfies the concavity condition.
    detrend : bool, optional (default=True)
        If True, applies OLS detrending to remove linear trends from the series,
        ensuring that the wavelet analysis focuses on the variance structure.
    plot_detrended : bool, optional (default=False)
        If True, plots the original and detrended series for visual inspection.
    plot_spectrum : bool, optional (default=False)
        If True, plots the wavelet spectral density analysis.
    plot_label : str, optional (default="Data")
        Extension to the title for the plot of the wavelet spectral density:
            title = "Wavelet Spectral Density" + plot_label
        This may be set to specify on which series the analysis is performed
        (e.g. "Wavelet Spectral Density - Residuals") and saving the plots
        with different file names.
    savefig : str, optional (default=None)
        Path to folder where to save the produced figures.
        If set to 'None', the figures will not be saved.
        
    Returns
    -------
    optimal_scale : int
        The estimated wavelet optimal scale (the chosen peak in the spectral
        density spectrum).
    mean_wsd : np.ndarray
        The computed wavelet spectral density values for each scale.
    mean_wsd_smoothed : np.ndarray
        The smoothed wavelet spectral density values for each scale.
    scales : np.ndarray
        The scale values corresponding to different window sizes.

    Raises
    ------
    TypeError
        If the input series is not a valid list, numpy array, pandas Series, or 
        single-column DataFrame.

    Notes
    -----
    - The function dynamically adapts to the length of the series to determine 
      a meaningful range of window sizes.
    - The **wavelet transform** (using the Morlet wavelet `'morl'`) is chosen because
      of its ability to capture localized variance patterns.
    - The **peak selection** ensures that the method avoids spurious peaks 
      or artifacts at large window sizes, by enforcing a concavity check 
      (d^2S/dT^2 < 0) and a fallback if none are found.

    Examples
    --------
    >>> import numpy as np
    >>> series = np.random.randn(100)  # Simulated time series
    >>> ow, oscale, mean_wsd, scales = estimate_window(series, peak_mode='first')
    >>> print(ow, oscale)
    6 12
    """
    # Step 0: Validate and prepare input
    series = prepare_series_input(series)
    max_window = len(series)

    # Step 1: Optional linear detrend
    if detrend:
        stationary_series = remove_trend_ols(series)
    else:
        stationary_series = series

    if plot_detrended:
        plt.figure(figsize=(8, 4))
        plt.plot(series, label="Original", c='C0', alpha=0.8)
        plt.plot(stationary_series, label="Detrended", c='C3', alpha=0.8)
        plt.title("Detrended Time Series - " + plot_label, fontsize=12)
        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend()
        plt.show()

    # Step 2: Compute Wavelet Spectral Density
    scales = np.arange(1, max_window)
    coefficients, _ = pywt.cwt(stationary_series, scales, 'morl')
    wsd = np.abs(coefficients) ** 2
    mean_wsd = wsd.mean(axis=1)  # average power at each scale

    # Step 3: Select optimal scale from wavelet spectrum
    # Using the "peak_selection" function that checks concavity
    optimal_scale, mean_wsd_smoothed = peak_selection(mean_wsd, scales, peak_mode=peak_mode)

    # Step 4: (Optional) Plot wavelet spectral density + chosen scale
    if plot_spectrum:
        with mpl.rc_context(rc=plot_style):
            sns.set_theme(style="darkgrid")
            mpl.rcParams.update(plot_style)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

            ax1.set_title("Series - " + plot_label)
            ax1.plot(series, label="Original", c='C0', alpha=0.8)
            ax1.set_xlabel("timestep")
            ax1.set_ylabel("value")
            
            ax2.set_title("Wavelet Spectral Density - " + plot_label)
            ax2.plot(scales, mean_wsd, marker='o', ms=5, alpha=0.5,
                     c='C0', label="Spectrum")
            ax2.plot(scales, mean_wsd_smoothed, marker='o', ms=5, alpha=0.8,
                     c='C3', label="Spectrum (smoothed)")
            ax2.axvline(optimal_scale, color='r', linestyle='--',
                        label=f"Optimal scale = {optimal_scale}", zorder=-1)
            ax2.set_xlabel("scale")
            ax2.set_ylabel("average power")
            
            plt.tight_layout()
            plt.legend()

            if savefig is not None:
                plt.savefig(savefig+'/optimal_scale_'+plot_label.replace(" ", "_")+'.png',\
                            dpi=600, bbox_inches='tight')            
            plt.show()


    return optimal_scale, mean_wsd, mean_wsd_smoothed, scales
