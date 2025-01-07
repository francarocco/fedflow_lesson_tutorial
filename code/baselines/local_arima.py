import pickle
import numpy as np
import itertools

import tensorflow as tf

from statsmodels.tsa.arima.model import ARIMA

import sys
sys.path.append('../')

from utilities.utilities import *


n_steps_in = 9
n_steps_out = 3
n_features = 6 # 6 considered features
num_clients = 5
epochs = 100
splits = range(5)
path_files = '../../data/input_data'

for split in splits:
    for i in range(num_clients):
        print(f'******Opening data of client {i} for split {split}******', flush = True)
        path_X_val = f'{path_files}/xval_{i}_split{split}_arima.pkl'
        with open(path_X_val, 'rb') as file:
            X_val = pickle.load(file)
        path_X_train = f'{path_files}/xtrain_{i}_split{split}_arima.pkl'
        with open(path_X_train, 'rb') as file:
            X_train = pickle.load(file)   
        path_X_test = f'{path_files}/xtest_{i}_split{split}_arima.pkl'     
        with open(path_X_test, 'rb') as file:
            X_test = pickle.load(file)


        ts = pd.Series(X_val[:n_steps_in])
        
        # Perform the test
        p_value = adf_test(ts)

        # If the series is not stationary (p-value > 0.05), take the first difference and test again
        d = 0
        while p_value > 0.05:
            ts = ts.diff().dropna()
            p_value = adf_test(ts)
            d += 1

        print(f'Differencing order d: {d}')

        p = range(0, 5) 
        q = range(0, 5) 
        d = [d]

        # Generate all different combinations of p, d, q triplets
        pdq = list(itertools.product(p, d, q))
        # Find the best combination based on AIC
        best_aic = float('inf')
        best_pdq = None

        for param in pdq:
            try:
                model = ARIMA(ts, order=param)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_pdq = param
            except:
                continue

        print(f'Best ARIMA parameters: {best_pdq} with AIC: {best_aic}', flush=True)
        
        
        
        # previsione
        forecasts, actuals = make_arima_forecasts(X_val,X_test, n_steps_in, n_steps_out,best_pdq)
        # valutazione dei risultati
        AE, SE = evaluate_arima_forecasts(actuals, forecasts, n_steps_in, n_steps_out)
        APE = evaluate_arima_MAPE_forecasts(actuals, forecasts, n_steps_in, n_steps_out)


        print('Saving model and results',flush=True)

        path_AE = f'{path_files}/results/AE_local_arima_{i}_split{split}.pkl'
        AE.to_pickle(path_AE)

        path_SE = f'{path_files}/results/SE_local_arima_{i}_split{split}.pkl'
        SE.to_pickle(path_SE)
        
        path_APE = f'{path_files}/results/APE_local_arima_{i}_split{split}.pkl'
        APE.to_pickle(path_APE)