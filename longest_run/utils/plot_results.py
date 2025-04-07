import numpy as np # type: ignore
from scipy import stats #type: ignore
from matplotlib import pyplot as plt #type: ignore
from matplotlib import rcParams #type: ignore

def plot_results(
    df_x_0, df_yhat, df_y, df_ysm, df_ysm_std, df_ysm_std_comm,
    df_yhat_comm, df_y_comm, df_stats, M_r_comm_n_dt, timesteps_comm,
    burn_in, win_local, t_LR, idx_LR, LR, stat_LR, p_LR, MAX_timesteps,
    base_metric, base_score_LR, method, chi2nu_thresh, rho_thresh, alpha,
    idx_slider=None, idx_t_slider=None, savefig=None):

    # Extracting convenient quantities:
    y_comm, yhat_comm = df_y_comm.values.flatten(), df_yhat_comm.values.flatten()
    r_comm = y_comm - yhat_comm
    r_comm_n = r_comm / df_ysm_std_comm.values.flatten()  # standardized residuals
    r_comm_n_dt_slider = M_r_comm_n_dt[idx_slider]

    # Check if signal has been smoothed:
    smoothed = not df_y.equals(df_ysm)

    # Slider window start
    idx_t_slider_min = max(-1, idx_t_slider - win_local)

    def calculate_bins(vmin, vmax, n=10):
        width = (vmax - vmin) / n
        return np.arange(vmin, vmax + width, width)

    # Bins for series values and residuals:
    bins = calculate_bins(min(y_comm.min(), yhat_comm.min()), max(y_comm.max(), yhat_comm.max()))
    bins_r = calculate_bins(r_comm.min(), r_comm.max())

    # Extract windowed quantities
    sl = slice(max(0, idx_slider - win_local), idx_slider)
    y_comm_dt_slider = y_comm[sl]
    yhat_comm_dt_slider = yhat_comm[sl]
    r_comm_dt_slider = r_comm[sl]
    std_r = np.std(r_comm_dt_slider, ddof=1)

    p_value_t = df_stats.loc[df_stats['idx_t'] == idx_t_slider, 'p_value'].values[0]
    stat_t    = df_stats.loc[df_stats['idx_t'] == idx_t_slider, 'stat'].values[0]

    # Create subplot grid
    fig, axes = plt.subplots(5, 2, figsize=(10, 6),
        gridspec_kw={'height_ratios': [3, 1, 1, 1, 1], 'width_ratios': [5, 1]})
    plt.subplots_adjust(wspace=0.03, hspace=0.08)

    # ---- Row: series ----------------------------------------------------
    ax = axes[0, 0]
    ax.axvspan(idx_t_slider_min + 1, idx_t_slider, color='blueviolet', alpha=0.15, label='slider range')
    
    # Full timesteps:
    if df_x_0 is not None:
        ax.plot(df_x_0.index, df_x_0['value'], c='C0', label='prompt window')
    ax.plot(df_yhat.index, df_yhat['value'], c='C2', label='$\hat{y}$')

    if smoothed:
        ax.plot(df_y.index, df_y['value'], c='C3', label='$y$', zorder=-1)
        ax.plot(df_ysm.index, df_ysm['value'], c='salmon', label='$y$ (smooth)')
        y1 = df_y['value'] - df_ysm_std['value']
        y2 = df_y['value'] + df_ysm_std['value']
        ax.fill_between(df_y.index, y1, y2, where=y1 < y2, color='salmon', alpha=0.3)
    else:
        ax.plot(df_y.index, df_y['value'], c='C3', label='$y$')

    # Common timesteps:
    ax.scatter(df_yhat_comm.index, df_yhat_comm['value'], c='C2', s=10)
    ax.scatter(df_y_comm.index, df_y_comm['value'], c='C3', s=10)

    ax.set_ylabel('series value')
    ax.tick_params(labelbottom=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=5)

    if df_x_0 is not None:
        padding = abs(np.min(df_x_0.index) - ax.get_xlim()[0])
        # padding automatically added by pyplot at the left edge of the x-axis
        # (to be added to the right edge, just after `MAX_timesteps`)
        ax.set_xlim(None, MAX_timesteps + padding)
        # limiting MAX timestep to `MAX_timesteps` for more clear visualization

    x_min, x_MAX = ax.get_xlim()
    y_min, y_MAX = ax.get_ylim()

    # ---- Row: histogram (series) ----------------------------------------
    hist_ax = axes[0, 1]
    hist_ax.clear()
    hist_ax.set_title('Histograms\nwithin slider', y=0.95, fontsize=8)

    # - reference histograms:
    counts_y, *_ = hist_ax.hist(yhat_comm[:idx_LR], bins=bins, orientation='horizontal',
                                density=True, histtype='step', color='C2', alpha=0.5)
    counts_yhat, *_ = hist_ax.hist(y_comm[:idx_LR], bins=bins, orientation='horizontal',
                                   density=True, histtype='step', color='C3', alpha=0.5)

    # - slider histograms:
    if idx_LR > 0:
        hist_ax.hist(yhat_comm_dt_slider, bins=bins, orientation='horizontal', density=True, color='C2', alpha=0.5)
        hist_ax.hist(y_comm_dt_slider, bins=bins, orientation='horizontal', density=True, color='C3', alpha=0.5)
        hist_ax.set_xlim(0, max(counts_y.max(), counts_yhat.max()) * 1.1)
        hist_ax.set_ylim(y_min, y_MAX)
    else:
        hist_ax.set_visible(False)

    hist_ax.tick_params(labelleft=False)
    hist_ax.set_xticks([])
    for spine in ['top', 'bottom', 'right']:
        hist_ax.spines[spine].set_visible(False)

    # ---- Row: residuals -------------------------------------------------
    ax = axes[1, 0]
    ax.axvspan(idx_t_slider_min + 1, idx_t_slider, color='blueviolet', alpha=0.15)
    ax.axhline(y=0, lw=0.5, ls='-', c='grey')  # Zero line
    ax.plot(df_yhat_comm.index, r_comm, c='goldenrod', alpha=0.8)
    ax.set_ylabel('res.')
    y_min, y_MAX = ax.get_ylim()

    # Histogram of residuals
    xx = np.linspace(-3 * std_r, 3 * std_r, 100)
    hist_ax = axes[1, 1]
    hist_ax.clear()
    hist_ax.plot(stats.norm.pdf(xx, 0, np.std(r_comm[:idx_LR], ddof=1)), xx, c='darkgoldenrod', alpha=0.5)
    counts_r, *_ = hist_ax.hist(r_comm[:idx_LR], bins=bins_r, orientation='horizontal',
                                density=True, histtype='step', color='goldenrod', alpha=0.5)
    if idx_LR > 0:
        hist_ax.plot(stats.norm.pdf(xx, 0, std_r), xx, c='darkgoldenrod')
        hist_ax.hist(r_comm_dt_slider, bins=bins_r, orientation='horizontal', density=True, color='goldenrod', alpha=0.5)
        hist_ax.set_xlim(0, counts_r.max() * 1.1)
        hist_ax.set_ylim(y_min, y_MAX)
    else:
        hist_ax.set_visible(False)

    hist_ax.tick_params(labelleft=False)
    hist_ax.set_xticks([])
    for spine in ['top', 'bottom', 'right']:
        hist_ax.spines[spine].set_visible(False)

    # ---- Row: standardized residuals ------------------------------------
    ax = axes[2, 0]
    ax.text(0.1, 0.6,
        '$\\langle\\sigma\\rangle$ estimated\nglobally', ha='center', va='center',
        transform=ax.transAxes, fontsize=8,
        bbox=dict(facecolor='seashell', edgecolor='none', alpha=0.8, boxstyle="round,pad=0.3"))
    ax.axvspan(idx_t_slider_min + 1, idx_t_slider, color='blueviolet', alpha=0.15)
    ax.axhline(y=0, lw=0.5, ls='-', c='grey')
    ax.plot(df_yhat_comm.index, r_comm_n, c='goldenrod', alpha=0.8)
    ax.set_ylabel('$\\frac{res}{\\langle\\sigma\\rangle}$', fontsize=14)
    if smoothed:
        # 3-sigma limits:
        ax.axhspan(-3, 3, color='goldenrod', alpha=0.05, label='3-sigma limits')
        ax.set_yticks([-3, 3])
        ax.set_ylim(-5, 5)
    axes[2, 1].set_visible(False)

    # ---- Row: metric score ----------------------------------------------
    ax = axes[3, 0]
    ax.axvspan(idx_t_slider_min + 1, idx_t_slider, color='blueviolet', alpha=0.15)
    ax.plot(df_stats['idx_t'], df_stats['base_score'], c='grey', label='prompt window')
    ax.set_xlim(x_min, x_MAX)
    ax.set_ylabel(base_metric)
    ax.tick_params(labelbottom=False)
    axes[3, 1].set_visible(False)

    # ---- Row: p-values / stats ------------------------------------------
    ax = axes[4, 0]
    ax.axvspan(idx_t_slider_min + 1, idx_t_slider, color='blueviolet', alpha=0.15)
    rcParams['hatch.linewidth'] = 5.0
    ax.axvspan(0, timesteps_comm[(burn_in-1) - 1], color='grey', hatch='/', alpha=0.3, label='burn-in')
    # NOTE: The burn-in stops 1 timestep _before_ the end of `burn_in`.
    #       That is, when  `burn_in` datapoints are available, that
    #       estimate is considered reliable.
    ax.axvline(x=idx_LR, c='grey', ls='--', lw=2, alpha=0.5, label='LR location')
   
    label_text = ''
    if method in ['chi2', 'ttest_r', 'ttest', 'KS_r', 'KS', 'KS_r_ex', 'KS_ex']:
        ax.plot(df_stats['idx_t'], df_stats['p_value'], c='grey')
        ax.set_yscale('log')
        ax.set_ylabel('p-value')
        label_text = f"$p$: {p_LR:.2g} ($\\alpha$: {alpha:.2g})"
    elif method == 'chi2nu':
        ax.plot(df_stats['idx_t'], df_stats['stat'], c='grey')
        ax.axhspan(0, chi2nu_thresh, color='grey', alpha=0.15)  # N-sigma limits
        ax.set_ylim(0, 10)
        ax.set_ylabel('$\\chi^2_{\\nu}$')
        label_text = f"$\\chi^2_{{\\nu}}$: {stat_LR:.2g} (thresh: {chi2nu_thresh:.2g})"
    elif method == 'rho':
        ax.plot(df_stats['idx_t'], df_stats['stat'], c='grey')
        ax.set_ylabel('$\\rho$')
        label_text = f"$stat$: {stat_LR:.2g} (thresh: {rho_thresh:.2g})"

    ax.set_xlim(x_min, x_MAX)
    ax.set_xlabel('test data index')
    ax.legend(loc='upper right')
    ax.text(0.13, -0.05,
        f"$t_{{~LR}}$: {t_LR} $-$ {base_metric}: {base_score_LR:.2g} $-$ LR: {LR:.2g} | {label_text}",
        transform=fig.transFigure, ha='left', va='center',
        bbox=dict(facecolor='C0', alpha=0.1, boxstyle="round,pad=0.3"))

    # ---- Row 4 Right: test panel ----------------------------------------
    ax = axes[4, 1]

    if idx_LR > 0:
        ax.clear()
        if method == 'chi2':
            plot_chi2_on_ax(ax, len(r_comm_dt_slider), p_value_t, stat_t)
        elif method == 'chi2nu':
            plot_chi2nu_on_ax(ax, r_comm_n_dt_slider, idx_t_slider_min, idx_t_slider, stat_t)
        elif method in ['ttest_r', 'ttest']:
            plot_ttest_on_ax(ax, y_comm_dt_slider, yhat_comm_dt_slider, p_value_t, stat_t,
                x_min=-std_r if method == 'ttest_r' else y_comm.min(),
                x_MAX=std_r if method == 'ttest_r' else y_comm.max(),
                oneway=(method == 'ttest_r'))
        elif method in ['KS_r', 'KS']:
            plot_KS_on_ax(ax, y_comm_dt_slider, yhat_comm_dt_slider, p_value_t,
                x_min=None if method == 'KS_r' else y_comm.min(),
                x_MAX=None if method == 'KS_r' else y_comm.max(),
                oneway=(method == 'KS_r'))
        elif method == 'rho':
            plot_rho_on_ax(ax, y_comm_dt_slider, yhat_comm_dt_slider, p_value_t, stat_t)
        elif method in ['KS_r_ex', 'KS_ex']:
            # NOTE: Plotting here would require a lot of data propagation from
            #       `ks_2samp_ex`, use `debug` option instead
            ax.set_visible(False)
    else:
        ax.set_visible(False)

    # ---- Final styling --------------------------------------------------
    for i in range(1, 3):
        axes[i, 0].set_xlim(x_min, x_MAX)
    for ax in axes.flatten():
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color('grey')

    if savefig is not None:
        plt.savefig(savefig+'/LR.png', dpi=600, bbox_inches='tight')
    plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def plot_chi2_on_ax(ax, dof, p_value, stat):

    ax.set_title('$\\chi^2$ PDF with %s dof' % dof, y=0.95, fontsize=8)

    # PDF:
    xx = np.linspace(stats.chi2.ppf(0.01, dof), stats.chi2.ppf(0.999, dof), 100)
    chi2_pdf = stats.chi2.pdf(xx, dof)
    ax.plot(xx, chi2_pdf, 'grey')

    # Location of observed chi2:
    ax.axvline(x=stat, c='goldenrod', alpha=0.8)
       
    ax.text(1, -0.9,
        str('$\\bar{\\chi}^2$: %.2g |  $p$: %.2g' % (stat, p_value)),
        horizontalalignment='right', verticalalignment='center',
        transform=ax.transAxes)#, bbox=dict(facecolor='white', alpha=1))
    
    ax.set_xlabel('$\\chi$')

    ax.tick_params(axis='y', which='both', bottom=False, left=False,
                right=True, labelbottom=False, labelleft=False, labelright=True)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def plot_chi2nu_on_ax(ax, r_comm_n_dt_slider, idx_t_slider_min, idx_t_slider, stat):

    ax.set_title('Res. [$\\sigma$] in slider', y=0.95, fontsize=8)
    ax.plot(np.arange(idx_t_slider_min, idx_t_slider), r_comm_n_dt_slider, c='goldenrod', label='res. [$\\sigma$]')
    #
    # Zero line:
    ax.axhline(y=0, lw=0.5, ls='-', c='grey')
    ax.text(1, -0.9,
        str('$\\bar{\\chi_{\\nu}}^2$: %.2g' % stat),
        horizontalalignment='right', verticalalignment='center',
        transform=ax.transAxes)#, bbox=dict(facecolor='white', alpha=1))
    
    ax.set_xlabel('timestep')

    ax.tick_params(axis='y', which='both', bottom=False, left=False,
                right=True, labelbottom=False, labelleft=False, labelright=True)
#------------------------------------------------------------------------------

#----------------------------------------------------------------------------
def plot_ttest_on_ax(ax, data_y, data_yhat, p_value, stat, x_min=None, x_MAX=None,
                     oneway=True):
    '''
    Displays on an existing plot axis the mean value of the data and comparison
    gaussian centered on 0 (if one-way t-test), or mean value of the data and
    predictions (if two-way t-test), along with their standard deviations.
    The data represent the values collected up to timestep `t`.
    This function does not actually perform the t-test, but just plots the
    mean values for diagnostic purposes, and reports the results (passed as
    an input) of a test which has been run outside the function.
    The one-way t-test has supposedly been run on the residuals, while the
    two-way t-test has been run on the raw data and predictions.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis object
        Axis on which the curves shall be displayed. 
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
    x_min, x_MAX : float (default: None)
        The min/MAX values to limit the axis x-range.
        If left to 'None', the range is deduced from the input data.
    oneway : bool, optional (default: True)
        Is it a one-way (as opposed to two-way) t-test?
    '''
    
    if oneway:
        resid = data_y - data_yhat
        std_resid = np.std(resid, ddof=1)
        std_mu_resid = std_resid / np.sqrt(len(resid))

        if x_min is None: x_min = min(min(resid), -3*std_resid)
        if x_MAX is None: x_MAX = max(max(resid), +3*std_resid)
        xx = np.linspace(x_min, x_MAX, 100)

        # Location of observed mean:
        ax.axvline(x=np.mean(resid), c='goldenrod', alpha=0.8)

        # Comparison distribution:
        gaussian_pdf = stats.norm.pdf(xx, 0, std_resid)
        ax.plot(xx, gaussian_pdf, c='darkgoldenrod')
        ax.plot([0., 0.], [0., stats.norm.pdf(0, 0, std_resid)], '--', c='grey')

        # Stdev on the mean segment:
        ax.fill_between(xx, gaussian_pdf, where=(xx >= -std_mu_resid) & (xx <= std_mu_resid), color='grey', alpha=0.5)

        ax.set_title('Resid $\\mu$ within slider', y=0.95, fontsize=8)
        ax.set_xlabel('residuals')

    if not oneway:
        mu_y    = np.mean(data_y)
        mu_yhat = np.mean(data_yhat)
        std_mu_y    = np.std(data_y,    ddof=1) / np.sqrt(len(data_y))
        std_mu_yhat = np.std(data_yhat, ddof=1) / np.sqrt(len(data_yhat))

        if x_min is None: x_min = min(min(data_y), min(data_yhat))
        if x_MAX is None: x_MAX = max(max(data_y), max(data_yhat))

        # Location of observed means:
        ax.axvline(x=mu_y,    c='C3', alpha=0.5)
        ax.axvline(x=mu_yhat, c='C2', alpha=0.5)

        # Stdev on the mean segment:
        ax.axvspan(mu_y-std_mu_y,       mu_y+std_mu_y,       color='C3',  alpha=0.3)
        ax.axvspan(mu_yhat-std_mu_yhat, mu_yhat+std_mu_yhat, color='C2', alpha=0.3)

        ax.set_title('Mean values in slider', y=0.95, fontsize=8)
        ax.set_xlabel('series value')

    ax.set_xlim(x_min, x_MAX)
    ax.tick_params(axis='y', which='both', bottom=False, left=False,
                right=True, labelbottom=False, labelleft=False, labelright=True)    

    ax.text(1, -0.9,
        str('$t$: %.2g |  $p$: %.2g' % (stat, p_value)),
        horizontalalignment='right', verticalalignment='center',
        transform=ax.transAxes)#, bbox=dict(facecolor='white', alpha=1))
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def plot_KS_on_ax(ax, data_y, data_yhat, p_value, x_min=None, x_MAX=None,
                  oneway=True):
    '''
    Displays on an existing plot axis the cumulative distribution of the data
    and a gaussian centered on 0 (if one-way K-S), or the cumulative
    distribution of the data and of the predictions (if two-way K-S), along
    with the location of their maximum distance (D; which is effectively the
    K-S statistics).
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
    ax : matplotlib.pyplot.axis object
        Axis on which the curves shall be displayed. 
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
    x_min, x_MAX : float (default: None)
        The min/MAX values to limit the axis x-range.
        If left to 'None', the range is deduced from the input data.
    oneway : bool, optional (default: True)
        Is it a one-way (as opposed to two-way) K-S?
    '''

    if oneway:
        resid = data_y - data_yhat
        std_resid = np.std(resid, ddof=1)
        
        if x_min is None: x_min = min(min(resid), -3*std_resid)
        if x_MAX is None: x_MAX = max(max(resid), +3*std_resid)

        bins = np.linspace(x_min, x_MAX, 1000)
        xx = (bins[1:] + bins[:-1]) / 2.0

        cdf1, _, _ = ax.hist(resid, bins=bins, density=True, cumulative=1, 
                                    histtype="step", color='goldenrod')
        cdf2 = stats.norm.cdf(xx, 0, std_resid)
        ax.plot(xx, cdf2, color='darkgoldenrod')
        # CDF of the gaussian centered at 0, and with stdev equal to the one of the
        # residuals   
        
        # where is the maximum difference between the model and sample CDFs?
        where_max_d = np.argmax(np.abs(cdf1 - cdf2))

        # find the height of the CDFs at their furthest point
        D1, D2 = cdf1[where_max_d], cdf2[where_max_d]

        # the maximum difference
        D = abs(D1 - D2)

        ax.set_title('KS within slider', y=0.95, fontsize=8)
        ax.plot([xx[where_max_d]]*2, [D1, D2], "k:")

    if not oneway:
        if x_min is None: x_min = min(min(data_y), min(data_yhat))
        if x_MAX is None: x_MAX = max(max(data_y), max(data_yhat))

        bins = np.linspace(x_min, x_MAX, 1000)
        xx = (bins[1:] + bins[:-1]) / 2.0

        cdf1, _, _ = ax.hist(data_y, bins=bins, density=True, cumulative=1, 
                                    histtype="step", color='C3')
        cdf2, _, _ = ax.hist(data_yhat, bins=bins, density=True, cumulative=1, 
                                    histtype="step", color='C2')
        
        # where is the maximum difference between the model and sample CDFs?
        where_max_d = np.argmax(np.abs(cdf1 - cdf2))

        # find the height of the CDFs at their furthest point
        D1, D2 = cdf1[where_max_d], cdf2[where_max_d]

        # the maximum difference
        D = abs(D1 - D2)

        ax.set_title('KS within slider', y=0.95, fontsize=8)
        ax.plot([xx[where_max_d]]*2, [D1, D2], "k:")

    ax.text(1, -0.9,
        str('$D$: %.2g |  $p$: %.2g' % (D, p_value)),
        horizontalalignment='right', verticalalignment='center',
        transform=ax.transAxes)#, bbox=dict(facecolor='white', alpha=1))

    ax.set_xlim(x_min, x_MAX)
    ax.set_xlabel('series value')

    ax.tick_params(axis='y', which='both', bottom=False, left=False,
                right=True, labelbottom=False, labelleft=False, labelright=True)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def plot_rho_on_ax(ax, data_y, data_yhat, p_value, stat, data_min=None, data_MAX=None):
    '''
    Displays on an existing plot axis the cumulative distribution of the data
    and predictions, and the location of their maximum distance (D; which is
    effectively the K-S statistics).
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
    ax : matplotlib.pyplot.axis object
        Axis on which the curves shall be displayed. 
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
    x_min, x_MAX : float (default: None)
        The min/MAX values to limit the axis x-range.
        If left to 'None', the range is deduced as the min/MAX values within
        `data_y` and `data_yhat`
    '''
    if data_min is None: data_min = min(min(data_y), min(data_yhat))
    if data_MAX is None: data_MAX = max(max(data_y), max(data_yhat))
    # data_min = 0
    # data_MAX = 1
    ax.scatter(data_y, data_yhat, s=5, color='grey')
    ax.axline((0, 0), slope=1, color='grey', linestyle='--')

    ax.set_title('Scatter plot within slider', y=0.95, fontsize=8)

    ax.text(1, -0.9,
        str('$\\rho$: %.2g | $p$: %.2g' % (stat, p_value)),
        horizontalalignment='right', verticalalignment='center',
        transform=ax.transAxes)#, bbox=dict(facecolor='white', alpha=1))

    ax_right = ax.twinx()
    # Hide frame around ax_right to avoid repetition of spines:
    for spine in ['top', 'bottom', 'left', 'right']:
        ax_right.spines[spine].set_visible(False)        

    ax_right.set_ylabel('$\hat{y}$')

    ax.set_xlim(data_min, data_MAX)
    ax.set_ylim(data_min, data_MAX)
    ax_right.set_ylim(data_min, data_MAX)
    ax.set_xlabel('$y$')

    ax.tick_params(axis='y', which='both', bottom=True, left=False,
                   right=False, labelbottom=False, labelleft=False, labelright=False)    
#----------------------------------------------------------------------------
