
import pandas as pd #type: ignore
import numpy as np #type: ignore
from scipy.signal import savgol_filter #type: ignore
import statsmodels.api as sm #type: ignore

def smooth_signal(df, smooth_algo='LOWESS', kernel_size=20, polyorder=3,
                  win_std=None,):
    
    '''Smooth a series and estimate the standard deviation around the smoothed
    model.
    
    The smoothing options, regulated by the parameter `smooth_algo`, are:
        - LOWESS (Locally Weighted Scatterplot Smoothing)
        - Savitzky-Golay polynomial filter
    The SG filter, while effective at smoothing and preserving higher-order
    features like peaks and troughs, applies a uniform set of convolution
    coefficients across the entire dataset, based on the chosen polynomial
    degree and window size. That means that the result may be significantly
    affected by the choice of `MAX_timesteps`.
    On the other hand, LOWESS is highly adaptive to local data variations
    because it fits lines/polynomials to localized subsets of data, with the
    fitting process being sensitive to the density and distribution of data
    points within each subset. Due to this larger adaptiveness, LOWESS is found
    to return a model whose residuals better estimate the local stochastic
    fluctuations.

    
    The standard deviation may be computed globally or locally.
    The two behaviors can be selected by picking a window size (`win_std`).
    In particular:
        - `win_std` == None
            selects the global calculation.
        - `win_std` == integer larger than 0
            selects the local calculation.
            A window size (`win_std`) of at least 20 is suggested in order to
            have enough datapoints for a solid estimate the local standard
            deviation.

    Caveat
        LOWESS slightly peaks in to the future when predicting the local model
        because of edge effects.
        So, a LOWESS model trained on all the data and then chopped within a
        given window slightly differs from a LOWESS model trained on the window
        itself.
        However, this does not represent an issue since the whole point of
        this smoothing is to evaluate the standard deviation after subtracting
        the "noiseless" signal. 
        What matters is that the noise standard deviation is evaluated in the
        exact same window in which the predictions are considered, because we
        do not want to leak noise across windows.
        In this way, the comparison is fair.
        If said "noiseless" signal is derived more consistently by also using
        the temporal future, that does not break this assumption. In that
        sense, we can consider the data and prediction as simultaneusly
        provided (as in fact they are, since the recursive prediction is
        pre-computed).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be smoothed.
    smooth_algo : string, optional (default: 'LOWESS')
        Algorithm used to derive the smooth model for the data `y`:
         - 'LOWESS': Locally Weighted Scatterplot Smoothing
         - 'SG':     Savitzky-Golay polynomial filter
    kernel_size : integer, optional (default: 5)
        Number of adjecent datapoints to be considered when deriving the local
        smooth model.
    polyorder : integer, optional (default: 3)
        Order of the polynomial to be fit.
        Only relevant when using the Savitzky-Golay filter.
    win_std : None or int (default: None)
        Size of the window used to compute the local standard deviation.
        If set to None, all points will be attributed the same standard
        deviation, derived using the model residuals from all data points.
    
    Returns
    -------
    df_sm : pd.DataFrame
        The smoothed dataframe.
    df_std : np.array
        Estimated standard deviation of each point [around the local polynomial
        model, if `win_std` is not None].
    '''
    
    data = df.values.flatten()
    
    if smooth_algo == 'SG':
        data_sm = savgol_filter(data, kernel_size, polyorder)
    if smooth_algo == 'LOWESS':
        n_points = len(data)
        frac = kernel_size / n_points
        X_ = np.arange(n_points)
        # pseudo-time indexing of `data`
        # (assumes all points are equally spaced; the actual values do not matter)
        result = sm.nonparametric.lowess(data, X_, frac=frac)
        X_sm_, data_sm = result[:, 0], result[:, 1]
        # smoothed X_ is discarded

    if win_std is None:
        data_std = np.full(len(data), np.std(data_sm - data, ddof=1))
    else:
        data_std = np.array([
            np.std(data[max(0, i - win_std // 2):min(len(data), i + win_std // 2 + 1)] - 
                data_sm[max(0, i - win_std // 2):min(len(data), i + win_std // 2 + 1)])
            for i in range(len(data))
        ])

    df_sm  = pd.DataFrame(data_sm,  columns=['value'])
    df_std = pd.DataFrame(data_std, columns=['value'])
    df_sm.index  = df.index
    df_std.index = df.index
    return df_sm, df_std