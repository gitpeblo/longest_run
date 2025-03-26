import numpy as np #type: ignore
import pandas as pd #type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #type: ignore
from prettytable import PrettyTable #type: ignore
from scipy import stats #type: ignore
from scipy.stats import ttest_1samp, ttest_ind, ks_1samp, ks_2samp, pearsonr #type: ignore
import scipy.stats as stats #type: ignore
import ipywidgets as widgets #type: ignore
from IPython.display import display #type: ignore
# src:
from .utils.array_to_pandas import array_to_pandas #type: ignore
from .utils.smooth_signal import smooth_signal #type: ignore
from .utils.recursive_predict import recursive_predict #type: ignore
from .utils.estimate_window import estimate_window #type: ignore
from .utils.plot_results import plot_results #type: ignore
from .utils.statistical_tests import ks_2samp_ex #type: ignore

def longest_run(model=None, x_0=None, y=None, y_roll=None, yhat_roll=None,
         method='KS',
         alpha=0.001, rho_thresh=0.95, chi2nu_thresh=5,
         base_metric='MAE', stride=1, n_forecast_iter='auto',
         smooth_algo='LOWESS', evaluation='local',
         burn_in=10, MAX_timesteps=None,
         kernel_size=None, polyorder=3,
         plot='all_widget', verbose=2, debug=False):
    '''
    Unified Longest Run (LR) metric based on statistical tests.

    This metric works by first identifying the amount of timesteps (`t_LR`),
    from the last known datum, after which the model predictions _first_
    diverge from the true data.
    The divergence is assessed by iteratively progressing along the time axis,
    each time collecting the data up to timestep "t" and executing a
    statistical test.
    Finally, the `LR` metric score itself is calculated by:
        1. Evaluating a `base_metric` (e.g., MAE) score over the
           data/predictions up to `t_LR` (`base_score_LR`).
        2. Normalizing (dividing) this score by `t_LR`.
    That is:
        `LR` = `base_score_LR` / `t_LR`

    Tests
        Given the predicted (`yhat`) and the true data (`y`), the available
        statistical tests (`method`) are:
            - chi2 ("chi2")
                Performs a chi2 test on the standardized residuals.
                The uncertainties on the residuals is estimated as the standard
                deviation around a smoothed model of the data.
                The smoothed model is created via a Savitzky–Golay filtering.
            - t-test on residuals ("ttest_r")
                Assesses the discrepancy between the mean of the residuals
                (`y` - `yhat`) and that of a normal distribution centered at 0
                and with standard deviation equal to that of the residuals
                themselves.
                The underlying idea is that, if the model is representative of
                the data, the residuals should be normally distributed.    
            - t-test ("ttest")
                Assesses the discrepancy between the means of `yhat` and `y`.
            - Kolmogorov-Smirnov test on residuals ("KS_r")
                Assesses the discrepancy between the residuals (`y` - `yhat`)
                and a normal distribution centered at 0 and with standard
                deviation equal to that of the residuals themselves.
                The underlying idea is that, if the model is representative of
                the data, the residuals should be normally distributed.
            - Kolmogorov-Smirnov test ("KS")
                Assesses the discrepancy between `yhat` and `y`.
            - Kolmogorov-Smirnov test ("KS_r_ex")
                Assesses the significance of the residuals (`y_sm` - `yhat`)
                where `y_sm` is a smoothed version of the data.
                It uses resampling to derive the exact PDF of the test
                statistic.
                Computationally slow.
            - Kolmogorov-Smirnov test ("KS_ex")
                Assesses the discrepancy between `yhat` and `y` using
                resampling to derive the exact PDF of the test statistic.
                Computationally slow.
            - Pearson's test ("rho") 
                Assesses the correlation between `yhat` and `y`.
                In practice the test is not actually performed, while the
                correlation coefficient is used directly, since it carries
                information about the direction of the correlation and in
                practice it spans less extreme ranges than the corresponding
                p-value.
        The `t_LR` is hence defined as the timestep at which the test's p-value
        or statistic falls below a pre-set threshold.
        In particular, the thresholds to reject the null hypothesis are:
            - chi2     | `alpha`: maximal p-value
            - ttest_r  | `alpha`: maximal p-value
            - ttest    | `alpha`: maximal p-value
            - KS_r     | `alpha`: maximal p-value
            - KS       | `alpha`: maximal p-value
            - KS_r_ex  | `alpha`: maximal p-value
            - KS_ex    | `alpha`: maximal p-value
            - rho      | `rho_thresh`: maximal correlation coefficient
            - chi2nu   | `chi2_thresh`: maximal chi2nu value
        The test is repeatedly executed from timestep 0 until `MAX_timesteps`
        (that is, it keeps scanning the data along the time axis independently
        on whether `t_LR` has already been identified).
    
    Timesteps
        A timestep is defined as the pseudo-time unit separating two consecutive
        datapoints inside a window.  It is assumed that all the data points are
        sampled with the same timestep.  The definition of timestep is not
        affected by the `stride`.

    Burin-n
        The local window starts at the current timestep, and extends back by
        `win_local` timesteps.
        That means that the first (`win_local`-1) timesteps contain less than
        the required number of datapoints for evaluating the statistical test.
        In order to guarantee at least a statistically significant amount of
        timesteps, the remaining points are padded by grabbing values from the
        future, from the current point up to `burn_in`, at most.
        
    Predictions
        The predicted values (`yhat`) are generated in a recursive fashion,
        i.e., the predicted values progressively get collated to the 'known
        data' to obtain subsequent predictions.
        The parameter `n_forecast_iter` regulates how many datapoints are
        forecasted at each recursion iteration.
    
    Caveat
        The `alpha` value may never be exceeded by the p-value, up to
        `MAX_timesteps`.  In that case, the LR corresponds to the last 
        timestep in the test set, indicating that, at the best of our
        knowledge, the model is compatible with the data up to that point.
    
    Parameters
    ----------
    model : sklearn-like model
        Pre-trained model.  Must contain a `predict` method.
    x_0 : array-like, shape (n_features)
        Prompt window to start the recurrent predictions.
    y : array-like, shape (n_samples, n_targets)
        True targets over which the metric shall be evaluated.
        It is composed of 'n_samples' of size 'n_targets' each.
        The targets may be partially overlapping  (i.e., if `stride` is smaller
        than `n_targets`).
        The first timestep should correspond to the first target timestep
        associated to the prompt window `x_0`.
    method : str, optional (default: 'KS')
        - 'chi2':    use the chi2 test
        - 'ttest_r': use the t-test on the residuals
        - 'ttest':   use the t-test to compare `y` and `yhat`
        - 'KS_r':    use the Kolmogorov-Smirnov test on the residuals
        - 'KS':      use the Kolmogorov-Smirnov test to compare `y` and `yhat`
        - 'KS_r_ex': use the Kolmogorov-Smirnov test on the residuals,
                     adopting the exact PDF for the statistic
        - 'KS_ex':   use the Kolmogorov-Smirnov test to compare `y` and `yhat`,
                     adopting the exact PDF for the statistic
        - 'rho':     use Pearson's correlation test
    alpha : float, optional (default: 0.005)
        Threshold for the K-S p-value to be considered significant.
        Ignored by other tests.
    rho_thresh : float, optional (default: 0.8)
        Threshold for the Pearson's correlation coefficient to be considered
        significant.
        Ignored by other tests.
    base_metric : str (default: 'MAE')
        Base metric to be measured at each timestep.  Options:
        - 'MAE'
        - 'SMAPE'
        - 'MSE'
        - 'RMSE'
        - 'r2'
    stride : int, optional (default: 1)
        Stride between consecutive windowed data (those in `X`), in units of
        data timesteps.
    n_forecast_iter : int or 'auto', optional (default: 'auto')
        Number of forecasting time steps to be predicted at each individual
        iteration.  These predictions will be stored in the output dataframe,
        and will rolled over in the predictor window for the next iteration.
        The number shall not exceed the number of target values that the model
        will output.
        If set to 'auto', it will be attriuted the number of expected target
        inferred from the model output.
    evaluation : str, optional (default: 'local')
        Regulates whether the statistical assessment shall be performed
        locally, or globally.
        The possible choices are:
        - 'local' :  Local evaluation
            The statistical assessment is performed over a striding window of
            size `win_local` timesteps.
            Every consecutive evaluation is therefore performed on a window of
            the same size.
        - 'global' : Global evaluation
            The statistical assessment is performed over a stretching window,
            starting from timestep 0, and progressively increasing in size.
            Every consecutive evaluation is therefore performed on a larger
            window.
    smooth_algo : string, optional (default: 'LOWESS')
        Algorithm used to derive the smooth model for the data `y`:
         - 'LOWESS': Locally Weighted Scatterplot Smoothing
         - 'SG':     Savitzky-Golay polynomial filter
    burn_in : int, optional (default: 10)
        Only relevant if `evaluation` = "local".
        Minimum number of timesteps to be considered for the evaluation
        of the statistical test.
    MAX_timesteps : int, optional (default: None)
        Number of future timesteps (beyond the last known datum) over which the
        LR shall be evaluated.
        If set to 'None', the maximum timestep corresponds to the inferred last
        timestep in `y` (after inverting the windowing).
    kernel_size : integer, optional (default: None)
        Number of adjecent datapoints to be considered when deriving the local
        smooth model.
        Only relevant if `method` = "chi2nu".
        If left to 'None', it will be estimated from the data.
    polyorder : integer, optional (default: 3)
        Order of the polynomial to be fit when using the Savitzky-Golay
        filtering.
        Only relevant if `method` = "chi2".
    plot : str, optional (default: 'all_widget')
        Option to plot intermediate and final results. Options include:
        - 'None': No plots.
        - 'results': Only final results.
        - 'results_widget': Only plot results, using a widget (for notebooks)
        - 'all': Plot both intermediate products and results
        - 'all_widget': Plot both intermediate products and results, results
             are plotted using a widget (for notebooks)
    
    Return
    ------
    result : dict
        Dictionary of results, containing:
        'df_stats' : pandas DataFrame
            Dataframe containing the p/values, KS statistics, and base score
            for each timestep.
        'stat_LR' : float
            The statistic corresponding to the LR timestep.
        'p_LR' : float
            The p-value corresponding to the LR timestep.
        't_LR' : int
            The timestep (counting from the first predicted datum) at which
            the LR is calculated.
        'base_score_LR' : float
            Score measured on the datapoints up to t_LR, using the desired
            input `metric`.
        'LR' : float
            The LR score.
        'base_metric_' : str
            Name of the user-requested base metric to utilize.

    }
    '''

    # Check if the user provided the rolled predictions (and test data)
    if y_roll is not None and yhat_roll is not None:
        if len(y_roll) != len(yhat_roll):
            raise ValueError("Rolled predictions and data have different lengths")

        user_provided_rolled = True
    else:
        user_provided_rolled = False

    if user_provided_rolled:
        # Irrelevant variabels for this case    
        n_targets = None
        n_forecast_iter =  None
        df_x_0 = None

        df_y    = pd.DataFrame(y_roll,    columns=['value'])
        df_yhat = pd.DataFrame(yhat_roll, columns=['value'])

    else:
        # Use model itself to infer the expected number of targets:
        try:
            n_targets = np.shape(model.predict([x_0]))[1]
        except:
            n_targets = 1

        if n_forecast_iter == 'auto': n_forecast_iter = n_targets
    
        df_x_0 = pd.DataFrame(data=x_0, columns=['value'],
                              index=np.arange(-len(x_0), 0))
    
        df_y = array_to_pandas(y, stride, t_offset=0)
        # NOTE: Temporarily setting the offset to 0 for easy comparison with the
        #       predicitons (will be changed later, for display purposes)
    
        if MAX_timesteps == None:
        # inferring `MAX_timesteps` from the last timestep in `df_y`
            MAX_timesteps = df_y.index.values[-1]
        
        df_yhat = recursive_predict(model, x_0, n_forecast_iter=n_forecast_iter,
                                    n_timesteps=MAX_timesteps)

    # Estimate optimal local window size --------------------------------------
    if evaluation == 'local':
        win_local, d, _, _ = estimate_window(df_y,\
                plot_spectrum=(plot == 'all' or plot == 'all_widget'))

        table = PrettyTable()
        table.title = str('Wavelet window detection')
        table.field_names = ['differencing (d)', 'win_local', 'adopted win_local']
        
        # Increase `win_local`, if it is less than burn-in
        if win_local < burn_in:
            table.add_row([str('%2s' % d), str('%s' % win_local), str('%s' % burn_in)])
            win_local = burn_in
        else:
            table.add_row([str('%2s' % d), str('%s' % win_local), str('%s' % win_local)])
        if verbose > 0: print(table)

    if kernel_size is None: kernel_size = win_local   
    #--------------------------------------------------------------------------
    
    if smooth_algo == 'SG' and kernel_size <= polyorder:
    # `kernel_size` must be larger than `polyorder` for Savitzky-Golay to work
        df_ysm = df_y
        df_ysm_std = pd.DataFrame(np.ones(len(df_ysm)), columns=['value'])
    else:
        df_ysm, df_ysm_std = smooth_signal(df_y, smooth_algo=smooth_algo,
                                        kernel_size=kernel_size, win_std=None)
        # NOTE: `win_std` is set to "None", hence the estimated std is
        #       calculated as the std across all the smooth model residuals
        #       throughout the whole data.
        #       This is NOT the local std!
    
    # Dataframes containing only common indexes:
    df_y_comm, df_yhat_comm = df_y.align(df_yhat, join='inner', axis=0)
    df_ysm_std_comm, _      = df_ysm_std.align(df_yhat, join='inner', axis=0)
    # ^using the estimated std and attributing it to the original data
    df_ysm_comm, _          = df_ysm.align(df_yhat, join='inner', axis=0)

    y_comm       = df_y_comm.values.flatten()
    yhat_comm    = df_yhat_comm.values.flatten()
    ysm_std_comm = df_ysm_std_comm.values.flatten()
    ysm_comm     = df_ysm_comm.values.flatten()

    timesteps_comm = df_y_comm.index.values
    # common timestep values (same as for `df_y_comm.index.values`)

    r_comm = y_comm - yhat_comm
    # residuals at common timesteps

    M_r_comm_n_dt = []
    # matrix of normalized residual arrays for every window dt <indexed by counter `j`>
    # NOTE: Store these data for `plot_results`

    # Looking for score value ------------------------------------------------
    df_stats = pd.DataFrame(columns=['idx_t', 'p_value', 'stat', 'base_score'])

    for idx_j, idx_t in enumerate(timesteps_comm):

        completion = (idx_j + 1) / len(timesteps_comm) * 100
        if verbose > 1: print(f'Completion: {completion:.2f}%', end='\r')        

        # > Values contained within time interval "delta"

        # global evaluation: `Delta` defined between timesteps 0 and `t`
        y_comm_Delta     = y_comm[:idx_j+1]
        yhat_comm_Delta  = yhat_comm[:idx_j+1]
        # Using the unique std value derived globally from the smooth model:     
        y_std_comm_Delta = ysm_std_comm[:idx_j+1]
        r_comm_Delta     = r_comm[:idx_j+1]        
        
        # local evaluation: `dt` defined between timesteps t-`win_local` and `t`
        y_comm_delta     = y_comm[max(0, idx_j+1-win_local):idx_j+1]
        yhat_comm_delta  = yhat_comm[max(0, idx_j+1-win_local):idx_j+1]
        ysm_comm_delta   = ysm_comm[max(0, idx_j+1-win_local):idx_j+1]
        # The std is re-calculated, to be the local std within dt:
        # NOTE: This is different than the one returned by `smooth_signal`
        y_std_comm_delta = np.full(len(y_comm_delta), np.std(ysm_comm_delta - y_comm_delta, ddof=1))
        r_comm_delta     = r_comm[max(0, idx_j+1-win_local):idx_j+1]

        # Select what to use in the remainder, based on evaluation
        if evaluation == 'global':
            y_comm_dt     = y_comm_Delta.copy()
            yhat_comm_dt  = yhat_comm_Delta.copy()
            y_std_comm_dt = y_std_comm_Delta.copy()
            r_comm_dt     = r_comm_Delta.copy()
        if evaluation == 'local':
            y_comm_dt     = y_comm_delta.copy()
            yhat_comm_dt  = yhat_comm_delta.copy()
            y_std_comm_dt = y_std_comm_delta.copy()
            r_comm_dt     = r_comm_delta.copy()

            # If idx_j is not large enought to cover the local window, grab the
            # missing data from the future
            if idx_j < burn_in:
                sl = slice(idx_j, burn_in)
                y_comm_dt[sl]     = y_comm_Delta[sl]
                yhat_comm_dt[sl]  = yhat_comm_Delta[sl]
                y_std_comm_dt[sl] = y_std_comm_Delta[sl]
                r_comm_dt[sl]     = r_comm_Delta[sl]


        # Independently of the "local"/"global" choice for the test, the base
        # score is evaluated from 0 up to timestep `t`:
        if base_metric == 'MAE':
            base_score = mean_absolute_error(y_comm[:idx_j+1], yhat_comm[:idx_j+1])
        if base_metric == 'SMAPE':
            base_score = smape(y_comm[:idx_j+1], yhat_comm[:idx_j+1])
        if base_metric == 'MSE':
            base_score = mean_squared_error(y_comm[:idx_j+1], yhat_comm[:idx_j+1])
        if base_metric == 'RMSE':
            base_score = np.sqrt(mean_squared_error(y_comm[:idx_j+1], yhat_comm[:idx_j+1]))
        if base_metric == 'r2':
            base_score = r2_score(y_comm[:idx_j+1], yhat_comm[:idx_j+1])

        r_comm_n_dt = r_comm_dt / y_std_comm_dt
        # standardized residuals
        M_r_comm_n_dt.append(r_comm_n_dt)

        if method == 'chi2':
            stat = np.sum(r_comm_n_dt**2) # chi2
            dof = len(r_comm_n_dt) # one per point
            p_value = stats.chi2.sf(stat, df=dof)
        if method == 'chi2nu':
            chi2 = np.sum(r_comm_n_dt**2)
            dof = len(r_comm_n_dt) # one per point
            stat = chi2 / dof # chi2nu
            p_value = np.nan
        if method == 'ttest_r':
            stat, p_value = ttest_1samp(r_comm_dt, 0)
        if method == 'ttest':
            stat, p_value = ttest_ind(y_comm_dt, yhat_comm_dt)
        if method == 'KS_r':
            gaussian_distr = stats.norm(0, np.std(r_comm_dt, ddof=1))
            stat, p_value = ks_1samp(r_comm_dt, gaussian_distr.cdf)
        if method == 'KS':
            stat, p_value = ks_2samp(y_comm_dt, yhat_comm_dt)
        if method == 'KS_r_ex':
            stat, p_value = ks_2samp_ex(y_comm_dt, yhat_comm_dt, method, t, debug=debug)
        if method == 'KS_ex':
            stat, p_value = ks_2samp_ex(y_comm_dt, yhat_comm_dt, method, t, debug=debug)
        if method == 'rho':
            corr, p_value = pearsonr(y_comm_dt, yhat_comm_dt)
            stat = corr
            # to stress where the stat comes from

        # debug: print all p-values for each timestep 
        if debug:       
            print("Method: %s | idx_t: %s | stat: %.4g | p-value: %.4g" %\
                    (method, idx_t, stat, p_value))

        df_stats.loc[len(df_stats)] = [idx_t, p_value, stat, base_score]

    # debug: plot CDFs        
    #plot_KS(y_comm_dt, yhat_comm_dt, t, p_value)

    df_stats = df_stats.astype({'idx_t': int})
    
    # Initialization - limit never violated:
    idx_violation = df_stats.index[-1] + 1
    # NOTE: +1 is added since "-1" is removed when `idx_LR` is created
    
    if method in ['chi2', 'ttest_r', 'ttest', 'KS_r', 'KS', 'KS_r_ex', 'KS_ex']:
        try:
            idx_violation = df_stats[df_stats['p_value'] < alpha].index[0]
            # index of corresponding to first violation of alpha      
        except:
            pass

    if method == 'chi2nu':
        try:
            idx_violation = df_stats[df_stats['stat'] > chi2nu_thresh].index[0]
            # index of corresponding to first violation of threshold      
        except:
            pass

    if method == 'rho':
        '''NOTE: For the Pearson test, we use the correlation rather than the
                 p-value, since the p-value drops too fast.
        '''
        try:
            idx_violation = df_stats[df_stats['stat'] < rho_thresh].index[0]
            # index of corresponding to first violation of threshold      
        except:
            pass

    if idx_violation == 0:
    #-special case: not even the first point is significant: returning index 0
        idx_LR = idx_violation
    else:
        idx_LR = idx_violation - 1

    p_LR    = df_stats['p_value'][idx_LR]
    stat_LR = df_stats['stat'][idx_LR]
    base_score_LR = df_stats['base_score'][idx_LR]

    idx_LR = int(df_stats.iloc[idx_LR]['idx_t'])
    # index of timestep corresponding to LR
    t_LR = idx_LR + 1
    # convert from index to number of timesteps

    LR = base_score_LR / (idx_LR+1)
    # LR score
    #-------------------------------------------------------------------------

    # Reporting metric -------------------------------------------------------
    table = PrettyTable()
    table.title = str('Values at t_LR')
    table.field_names = ['t_LR', base_metric, 'LR', 'method', 'stat', 'p-value']
    table.add_row([t_LR, str('%.2g' % base_score_LR),
                   str('%.2g' % LR),
                   str('%s' % method),
                   str('%.2g' % stat_LR),
                   str('%.2g' % p_LR)])
    if verbose > 0: print(table)
    #-------------------------------------------------------------------------

    # Plotting results -------------------------------------------------------    
    # Data necessary for plotting
    plot_data = {
        'df_x_0': df_x_0,
        'df_yhat': df_yhat,
        'df_y': df_y,
        'df_ysm': df_ysm,
        'df_ysm_std': df_ysm_std,
        'df_ysm_std_comm': df_ysm_std_comm,
        'df_yhat_comm': df_yhat_comm,
        'df_y_comm': df_y_comm,
        'df_stats': df_stats,
        'M_r_comm_n_dt': M_r_comm_n_dt,
        'timesteps_comm': timesteps_comm,
        'burn_in': burn_in,
        'win_local': win_local,
        't_LR': t_LR,
        'idx_LR': idx_LR,
        'LR': LR,
        'stat_LR': stat_LR,
        'p_LR': p_LR,
        'evaluation': evaluation,
        'MAX_timesteps': MAX_timesteps,
        'base_metric': base_metric,
        'base_score_LR': base_score_LR,
        'method': method,
        'chi2nu_thresh': chi2nu_thresh,
        'rho_thresh': rho_thresh,
        'alpha': alpha,
    }

    if plot in ['results', 'all']:
        
        plot_results(**plot_data, idx_slider=idx_LR, idx_t_slider=idx_LR)   
        
    if plot in ['results_widget', 'all_widget']:

        def update_histogram(idx_t_slider):
            idx_slider = np.where(df_stats['idx_t'].values == idx_t_slider)[0][0]
            # index of slider (corresponding to `y_comm` or `yhat_comm` indexing)

            plot_results(**plot_data, idx_slider=idx_slider, idx_t_slider=idx_slider)    

        slider = widgets.SelectionSlider(
            options=timesteps_comm[win_local:],
            value=t_LR,
            description='Probing $t$:',
            continuous_update=True,  # Update while the user is moving the slider?
            orientation='horizontal',
            readout=True  # Display the label of the selected option
        )    

        # Display the widget and output
        output = widgets.interactive_output(update_histogram, {'t_slider': slider})
        display(slider, output)
    #-------------------------------------------------------------------------

    # Preparing returned data structure --------------------------------------
    result = {
        'win_local': win_local,
        'df_stats': df_stats,
        'stat_LR': stat_LR,
        'p_LR': p_LR,
        't_LR': t_LR,
        'base_score_LR': base_score_LR,
        'LR': LR,
        'base_metric': base_metric,
    }
    #-------------------------------------------------------------------------

    return result


# SUPPORT FUNCTIONS ##########################################################

#----------------------------------------------------------------------------
def smape(y, yhat):
    '''
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between two
    arrays.
    '''
    denominator = (np.abs(y) + np.abs(yhat)) / 2.0
    diff = np.abs(y - yhat) / denominator
    diff[denominator == 0] = 0.0 
    # handle the case where both `y`` and `yhat` are 0
    return 100 * np.mean(diff)
#----------------------------------------------------------------------------

#############################################################################
