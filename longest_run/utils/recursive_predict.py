import numpy as np #type: ignore
import pandas as pd #type: ignore

def recursive_predict(model, x_0, n_forecast_iter='auto', n_timesteps=100):
    '''
    Performs recursive predictions given a model and a prompt window.
    Parameters
    ----------
    model : sklearn-like model
        Pre-trained model.  Must contain a `predict` method.
    x_0 : array-like, shape (n_features)
        Prompt window to start the recurrent predictions.
    n_forecast_iter : int or 'auto', optional (default: 'auto')
        Number of forecasting time steps to be predicted at each individual
        iteration.  These predictions will be stored in the output dataframe,
        and will rolled over in the predictor window for the next iteration.
        The number shall not exceed the number of target values that the model
        will output.
        If set to 'auto', it will be attriuted the number of expected target
        inferred from the model output.
    n_timesteps : int, optional (default: 100)
        Number of future timesteps (beyond the last known datum) to be
        predicted.
        
    Return
    ------
    df_yhat : pd.DataFrame, shape (n_timesteps)
        Dataframe of predictions.
    
    '''
    # Initializing predictor window to the prompt window:
    x_i = x_0
    
    # Use model itself to infer the expected number of targets:
    try:
        n_targets = np.shape(model.predict([x_0]))[1]
    except:
        n_targets = 1
    
    if n_forecast_iter is 'auto': n_forecast_iter = n_targets
    if n_forecast_iter > n_targets:
        print('WARNING: recursive_predict::')
        print('\tNumber of iteration forecasted timesteps larger than the model output')
        print('\t\t`n_forecast_iter` = %s, `n_targets` = %s' % (n_forecast_iter, n_targets))
        print('\tSetting `n_forecast_iter` to %s' % n_targets)
        n_forecast_iter = n_targets
        
    df_yhat = pd.DataFrame()
    
    # Rolling predictions:
    i = 0
    # iteration counter
    t = 0
    # timestep counter (latest prediction reached)
    while t < n_timesteps:

        yhat_i = model.predict([x_i])
        # NOTE: Labelling by `i` (and not `t`) because `yhat_i` contains
        #       `n_forecast_iter` values
        
        yhat_i_r = yhat_i.flatten()[: n_forecast_iter]
        # predictions to be stored at iteration i, and rolled over at
        # iteration i+1
        
        # Updating window:
        x_i = x_i[n_forecast_iter:]
        x_i = np.concatenate([x_i, yhat_i_r])
        # dropping first `n_forecast_iter` elements and collating predictions

        df_yhat_i = pd.DataFrame(data=yhat_i_r,
                         columns=['value'],
                         index=np.arange(t, t+n_forecast_iter, 1).astype(int))
        df_yhat_i.rename_axis('idx_t', inplace=True)
        
        df_yhat = pd.concat([df_yhat, df_yhat_i])
                            
        # Incrementing loop counter:
        i+=1
            
        # Incrementing timestep counter:
        t+=n_forecast_iter
        # NOTE: Every time a prediction is performed, the predictions
        #       are attached to the next `x_i` (regardless of the stride).
    
    return df_yhat

