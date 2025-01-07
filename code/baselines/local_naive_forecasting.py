import pickle
import numpy as np

import tensorflow as tf


import sys
sys.path.append('../')

from utilities.utilities import *



n_steps_in = 9
n_steps_out = 3
num_clients = 5
splits = range(5)

path_files = '../../data/input_data'

for split in splits:
    for i in range(num_clients): #working area by area
        path_scaler = f'{path_files}/scalers/scaler_{i}_split{split}.pkl'
        scaler = pickle.load(open(path_scaler, 'rb'))

        path_X_test = f'{path_files}/xtest_{i}_split{split}_naive_forecasting.pkl'     
        with open(path_X_test, 'rb') as file:
            test_values = pickle.load(file)

        # previsione
        forecasts = make_forecasts_naive_forecasting(test_values, n_steps_in, n_steps_out)
        AE, SE = evaluate_forecasts_naive_forecasting(test_values, forecasts, n_steps_in, n_steps_out)
        APE = evaluate_MAPE_forecasts_naive_forecasting(test_values, forecasts, n_steps_in, n_steps_out)

        print('Saving model and results',flush=True)

        path_AE = f'{path_files}/results/AE_naive_forecasting_{i}_split{split}.pkl'
        AE.to_pickle(path_AE)

        path_SE = f'{path_files}/results/SE_naive_forecasting_{i}_split{split}.pkl'
        SE.to_pickle(path_SE)
        
        path_APE = f'{path_files}/results/APE_naive_forecasting_{i}_split{split}.pkl'
        APE.to_pickle(path_APE)
