
import numpy as np #type: ignore
from matplotlib import pyplot as plt #type: ignore
from scipy import stats #type: ignore
from sklearn.pipeline import Pipeline #type: ignore
from sklearn.preprocessing import PolynomialFeatures #type: ignore
from sklearn.linear_model import LinearRegression #type: ignore
from scipy.stats import gaussian_kde #type: ignore
from scipy.stats import ks_2samp #type: ignore

#------------------------------------------------------------------------------
def ks_2samp_ex(y, yhat, method='KS_ex', t=None, debug=False):
    '''Perform an exact K-S test where the PDF of the statistic is empirically
    derived from the data, using a resampling technique.
    
    This differs from the standard K-S test which assumes i.i.d. data (and
    homoscedasticity, although this is not a concern if the test is run
    "locally"), and hence adopts an incorrect PDF for the test statistic. 
    '''

    if debug:
        print('Slider timestep: %s' % t)
    
    # > Perform local fit on `y`

    X_ = np.arange(len(y)).reshape(-1, 1)
    # pseudo-time indexing of `y`
    # (assumes all points are equally spaced; the actual values do not matter)

    model = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2)),
        ('linear_regression', LinearRegression())
    ])
    model.fit(X_, y)

    yyhat = model.predict(X_)
    # local model predictions at each `X_`
    rr = y - yyhat
    # residuals of data with respect to local model
    rrhat = yhat - yyhat
    # residuals of predictions with respect to local model
    #
    # NOTE: Double letters (e.g., "yy" or "rr" refer to quantities related to
    #       the local model)

    # > Draw samples for the PDF
    n_samp = 1000
    # number of mock samples (i.e., number of measurements of the test statistic)
    size = len(y)
    # the number of samples drawn from the parametrized distribution at each
    # iteration shall equal taht of the original datapoints
    factor = 1
    # factor used to expand the sampling distribution of residuals
    # (to be more conservative; not used)

    xx = np.linspace(X_.min(), X_.max(), size).reshape(-1, 1)

    rrs_samp = []
    ys_samp = []
    stats_y_samp = []
    stats_r_samp = []

    for _ in range(n_samp):

        # Sampling mu around actual residual mu:
        mu_samp = np.random.normal(
            loc=np.mean(rr), scale=factor*(np.std(rr, ddof=1)/np.sqrt(size)),
            size=1)[0]

        # Draw `size` samples from the normal distribution:
        rr_samp = np.random.normal(
            loc=mu_samp, scale=factor*np.std(rr, ddof=1), size=size)

        # Add trend back for test on raw data:
        y_samp = yyhat + rr_samp

        # Measure statistic for current sampling
        # - raw data:
        stat_y_samp = ks_2samp(y, y_samp).statistic
        # - residuals:
        stat_r_samp = ks_2samp(rr, rr_samp).statistic

        # Store:
        rrs_samp.append(rr_samp)
        ys_samp.append(y_samp)
        stats_y_samp.append(stat_y_samp)
        stats_r_samp.append(stat_r_samp)

        #debug: individual tests: plot_KS(y,  y_samp,  9.999, 9.999); plt.show()
        #debug: individual tests: plot_KS(rr, rr_samp, 9.999, 9.999); plt.show()

    # > Actual statistic measured in data
    if method == 'KS_ex':
        stat, _ = ks_2samp(y, yhat)
        stats_samp = stats_y_samp
    if method == 'KS_r_ex':
        stat, _ = ks_2samp(rr, rrhat)
        stats_samp = stats_r_samp
    # NOTE: This step is only exploting `ks_2samp` to obtain the statistic,
    #       since the actual p-value is obtained below using the exact PDF

    # > Find p-value
    # NOTE: It is easier to calculate this on a KDE-smoothed PDF, since the
    #       raw PDF is very quantized when few datasteps (e.g., 20) are used,
    #       and hence the standard deviation is underestimated.

    # Smooth with KDE:
    data_spread = np.max(stats_samp) - np.min(stats_samp)
    # NOTE: Experimentally better using this as kernel size for KDE, than using
    #       some multiple of the standard deviation
    kde = gaussian_kde(stats_samp, bw_method=data_spread)
    p_value = kde.integrate_box_1d(stat, np.inf)

    # > Find z-value (not used)

    # Find PDF width with quantiles:
    kde_samples = kde.resample(size=1000)
    q_50  = np.quantile(kde_samples, 0.5)
    # median
    q_upp = np.quantile(kde_samples, 0.8413)
    # upper quantile

    dq = q_upp - q_50
    # delta quantile: distance between median and upper quantile
    
    z_value = (stat - np.median(stats_samp)) / dq
    # z-value: how many "standard errors" away from median of the PDF `stat` is

    # > Plot
    if debug:
        fig, axes = plt.subplots(figsize=(6, 3), nrows=3, sharex=True)
        plt.suptitle('Sampling')

        # Data:
        xx = np.linspace(X_.min(), X_.max(), 100).reshape(-1, 1)

        axes[0].scatter(X_, y,    c='C3', label='y')
        axes[0].scatter(X_, yhat, c='C2', label='yhat')
        axes[0].plot(xx, model.predict(xx), c='tomato', label='local model')
        axes[0].set_ylabel('series')
        
        axes[1].plot(X_, rr, c='goldenrod', label='res.')
        for _, rr_samp in enumerate(rrs_samp[:10]):
            if _ == 0:
                axes[1].plot(X_, rr_samp, c='lightgrey', alpha=0.3, label='res. sampled (a few)')
            else:
                axes[1].plot(X_, rr_samp, c='lightgrey', alpha=0.3)
        axes[1].set_ylabel('resid')
        
        axes[2].scatter(X_, y,    c='C3', label='data')
        axes[2].scatter(X_, yhat, c='C2', label='yhat')
        for _, y_samp in enumerate(ys_samp[:10]):
            if _ == 0:
                axes[2].plot(X_, y_samp, c='tomato', alpha=0.3, label='y sampled (a few)')
            else:
                axes[2].plot(X_, y_samp, c='tomato', alpha=0.3)
        #
        axes[2].set_ylabel('series')
        axes[2].set_xlabel('pseudo-timestep')
        #
        for ax in axes:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.setp(axes[0].get_xticklabels(), visible=False)
        plt.setp(axes[1].get_xticklabels(), visible=False)
        # second command redundant since `sharex` == True
        plt.show()

        print('p:', p_value)

        # PDF:
        fig = plt.figure()
        xx_kde = np.linspace(np.min(stats_samp+[stat]), np.max(stats_samp+[stat]), 500)
        # range over which the KDE model shall be plotted
        kde_curve = kde(xx_kde)
        # empirical PDF modelled via KDE
        kde_curve_norm = kde_curve * len(stats_samp) * (max(stats_samp) - min(stats_samp)) / len(xx_kde)
        # empirical PDF, normalized to match the histogram

        plt.hist(stats_samp, density=True, color='C0', alpha=0.7, label='Empirical distribution')
        plt.axvline(x=q_50, ls='--', color='C0', label='Median')

        # Visualize where individual statistic measurements are:
        segment_height = kde_curve.max() * 0.1
        for v, value in enumerate(stats_samp):
            if v > 0:
                plt.plot([value, value], [0, segment_height], color='C0', alpha=1)
            else:
                plt.plot([value, value], [0, segment_height], color='C0', alpha=1, label='Raw sampled values')
        
        plt.axvline(x=stat, ls='-', lw=2, color='goldenrod', label='Measured value')

        plt.plot(xx_kde, kde_curve_norm, lw=2, color='firebrick', label='KDE')
        mask = (xx_kde >= q_50) & (xx_kde <= q_upp)
        plt.fill_between(xx_kde[mask], 0, kde_curve_norm[mask], color='firebrick', alpha=0.3, label='Upper quantile')

        plt.xlabel('KS statistic')
        plt.ylabel('Probability Density')
        plt.title('Empirical PDF of KS statistic')

        plt.legend()
        plt.show()        

    return stat, p_value
#------------------------------------------------------------------------------

#----------------------------------------------------------------------------
def plot_KS(data_y, data_yhat, t, p_value):
    '''
    Used for debug only.

    Plots the cumulative distribution of the data and predictions, and the
    location of their maximum distance (D; which is effectively the K-S
    statistics).
    The data represent the values collected up to timestep `t`.
    This function does not actually perform the K-S test, but just plots the
    cumulatives for diagnostic purposes.

    Caveat:
        The actual K-S p-value is calculated outside the function and it is
        passed as a parameter, only to be printed on the plot.

        For this reason, the maximum distance is _not exactly_ corresponding to
        the p-value, because it is calculated inside the function.

    Parameters
    ----------
    data_y : array-like, shape (n_samples)
        Values corresponding to the true data.
    data_yhat : array-like, shape (n_samples)
        Values corresponding to the predictions.
    t : int
        Index indicating up to which timestep the input `data_y` ( `data_yhat`)
        refer to.  Used only to be printed on the plot.
    p_value : float
        The p-value obtained by running a K-S test between `data_y` and 
        `data_yhat`.  Used only to be printed on the plot.
    '''

    bins = np.linspace(min(min(data_y), min(data_yhat)),
                       max(max(data_y), max(data_yhat)), 1000)
    xx = (bins[1:] + bins[:-1]) / 2.0

    plt.figure(figsize=(6, 3))
    plt.title('timestep: %s | KS test | p-value: %.3g' % (t, p_value))

    cdf1, _, _ = plt.hist(data_y, bins=bins, density=True, cumulative=1, 
                                histtype="step", color='C3', label='$y$')
    cdf2, _, _ = plt.hist(data_yhat, bins=bins, density=True, cumulative=1, 
                                histtype="step", color='C2', label='$\hat{y}$')
    
    # where is the maximum difference between the model and sample CDFs?
    where_max_d = np.argmax(np.abs(cdf1 - cdf2))

    # find the height of the CDFs at their furthest point
    D1, D2 = cdf1[where_max_d], cdf2[where_max_d]

    # the maximum difference
    D = abs(D1 - D2)

    plt.plot([xx[where_max_d]]*2, [D1, D2], "k:", label="D={:.3g}".format(D))
    plt.xlabel('value')
    plt.ylabel('CDF')
    plt.legend(loc="upper left")
    plt.show()
#----------------------------------------------------------------------------
