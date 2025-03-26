import numpy as np #type: ignore
import pandas as pd #type: ignore
import pywt #type: ignore
from statsmodels.tsa.stattools import kpss #type: ignore
from matplotlib import pyplot as plt #type: ignore
import matplotlib as mpl #type: ignore
import seaborn as sns #type: ignore
from scipy.signal import find_peaks, peak_prominences #type: ignore
import warnings #type: ignore

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

def find_differencing_order(series, max_d=5, significance_level=0.005):
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

    while d < max_d:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Convert warnings to exceptions
                _, p_value, _, _ = kpss(test_series, regression='ct', nlags="auto")
        except (RuntimeWarning, UserWarning):  # Catch numerical warnings including InterpolationWarning
            return d, test_series  # Assume stationarity if warning occurs
        except Exception:  # Catch unexpected failures
            break  # Stop if KPSS fails for any other reason

        if p_value > significance_level:
            return d, test_series  # Stationary w.r.t trend

        test_series = np.diff(test_series)
        d += 1

    return d, test_series

def refined_wavelet_peak_selection(mean_wsd, scales):
    """
    Identifies the most significant and stable peak in the wavelet spectral density, 
    avoiding edge-based artifacts and spurious peaks.

    The function analyzes the wavelet spectral density \( S(T) \) to determine the optimal 
    time scale at which variance stabilizes. It selects the most prominent peak that 
    represents a meaningful characteristic scale while ensuring smoothness and avoiding 
    boundary effects.

    Method
    ------
    - Identify all local maxima in the spectral density \( S(T) \).
    - Compute **prominence** for each peak to determine its relative importance.
    - Sort peaks based on prominence (most significant first).
    - Select the **most prominent, smooth peak** that satisfies:
        - It is **concave** (second derivative \( d^2S/dT^2 < 0 \)).
        - It is **not a spurious peak** from noise fluctuations.
    - If no smooth peak is found, fallback to the **most prominent peak** overall.

    Parameters
    ----------
    mean_wsd : np.ndarray
        The wavelet spectral density values corresponding to different scales.
    scales : np.ndarray
        The scale values (time window sizes) associated with the spectral density.

    Returns
    -------
    optimal_scale : float
        The refined optimal window size, representing the characteristic time scale 
        for stable variance estimation.

    Raises
    ------
    ValueError
        If the input arrays have inconsistent dimensions.

    Notes
    -----
    - The function avoids edge peaks that are artificially increasing near the
      maximum window size.
    - The **second derivative test** ensures that the selected peak is part of a 
      smoothly varying trend rather than a sharp noise artifact.
    - The **prominence-based sorting** ensures that truly dominant features are 
      prioritized.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import find_peaks, peak_prominences
    >>> mean_wsd = np.array([0.1, 0.3, 1.2, 0.9, 3.5, 5.0, 4.8, 0.5])
    >>> scales = np.arange(1, len(mean_wsd) + 1)
    >>> refined_window = refined_wavelet_peak_selection(mean_wsd, scales)
    >>> print(refined_window)
    6
    """

    # Compute second derivative (smoothness criterion)
    d2S_dT2 = np.diff(mean_wsd, n=2)

    # Identify all peaks in the spectral density
    peaks, _ = find_peaks(mean_wsd)

    if len(peaks) == 0:
        return scales[np.argmax(mean_wsd)]  # Fallback to global max if no peaks found

    # Compute peak prominences (significance of each peak)
    prominences = peak_prominences(mean_wsd, peaks)[0]

    # Sort peaks by prominence (most significant peaks first)
    sorted_peaks = sorted(zip(peaks, prominences), key=lambda x: -x[1])

    # Select the best peak: Most prominent, but also smooth (not a noisy spike)
    best_peak = None
    for peak, _ in sorted_peaks:
        if peak > 1 and peak < len(d2S_dT2) and d2S_dT2[peak - 1] < 0:  # Ensure concavity
            best_peak = peak
            break  # Stop at the first valid significant peak

    # If no smooth peak is found, fallback to the most prominent peak
    if best_peak is None:
        best_peak = sorted_peaks[0][0]

    return scales[best_peak]

def estimate_window(series, plot_spectrum=False):
    """
    Estimates the optimal rolling window size for variance stabilization using
    wavelet spectral density analysis. This function identifies the dominant
    time scale over which the variance structure remains stable.

    The method applies:
    - **KPSS test** to determine and remove deterministic trends.
    - **Wavelet Transform** to extract the dominant periodicity in local variance.
    - **Refined peak selection** to identify the most significant and stable scale.
    - **Visualization** to interpret the wavelet power spectrum.

    Relationship Between Wavelet Power Spectrum and Window Size
    -----------------------------------------------------------
    - The **wavelet power spectrum** represents the energy distribution of the
      time series across different scales (window sizes).
    - The higher the power at a given scale, the more variance is explained by
      that window size.
    - The **optimal window size** corresponds to the **peak of the power spectrum**, 
      indicating the characteristic timescale over which fluctuations stabilize.

    Parameters
    ----------
    series : array-like
        The input time series data, which can be a list, numpy array, pandas Series, 
        or single-column DataFrame.
    plot_spectrum : bool, optional, default=False
        If True, plots the wavelet spectral density analysis.

    Returns
    -------
    refined_optimal_window : int
        The estimated optimal window size for variance estimation.
    d : int
        The number of differences applied to achieve trend stationarity.
    mean_wsd : np.ndarray
        The computed wavelet spectral density values for each scale.
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
    - The **wavelet transform** (using the Morlet wavelet) is chosen because of its 
      ability to capture localized variance patterns.
    - The **refined peak selection** ensures that the method avoids spurious peaks 
      or artifacts at large window sizes.

    Examples
    --------
    >>> import numpy as np
    >>> series = np.random.randn(100)  # Simulated time series
    >>> optimal_window, d, mean_wsd, scales = estimate_window(series)
    >>> print(optimal_window, d)
    12 1
    """

    # Preprocess input
    series = prepare_series_input(series)
    max_window = len(series)
    
    # Step 1: Determine differencing order using KPSS
    d, stationary_series = find_differencing_order(series)

    # Step 2: Compute Wavelet Spectral Density
    scales = np.arange(1, max_window)
    coefficients, _ = pywt.cwt(stationary_series, scales, 'morl')
    wsd = np.abs(coefficients) ** 2
    mean_wsd = wsd.mean(axis=1)

    # Step 3: Apply refined peak selection method
    refined_optimal_window = refined_wavelet_peak_selection(mean_wsd, scales)

    # Step 4: Plot the refined spectral density analysis
    if plot_spectrum:
        with mpl.rc_context(rc=plot_style):  # Use a context manager for temporary style
            sns.set_theme(style="darkgrid")  # Apply seaborn theme
            mpl.rcParams.update(plot_style)  # Reapply custom settings in case seaborn overrode them
            
            plt.figure()
            plt.plot(scales, mean_wsd, marker='o', ms=5, alpha=0.8, label="Wavelet Spectral Density")
            plt.axvline(refined_optimal_window, color='r', linestyle='--',
                        label=f"optimal window = {refined_optimal_window}", zorder=-1)
            plt.title("Wavelet Spectral Density | differencing (d): " + str(d))
            plt.xlabel("scale (window size)")
            plt.ylabel("average power")
            plt.legend()
            plt.show()
        
    return refined_optimal_window, d, mean_wsd, scales
