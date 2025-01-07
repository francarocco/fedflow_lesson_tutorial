import pickle
import numpy as np

import tensorflow as tf

import sys
sys.path.append('../')

from utilities.utilities import *
from utilities.networks import * 


n_steps_in = 9
n_steps_out = 3
n_features = 6 # 6 considered features
num_clients = 5
epochs = 100
splits = range(5)

path_files = '../../data/input_data'

print(tf.config.list_physical_devices('GPU'),flush=True)

for split in splits:
    for i in range(num_clients): #working area by area
        print(f'**********Working on client {i}-th*************',flush=True)

        # reading datasets
        print('Reading training, validation, test set',flush=True)
        path_X_train = f'{path_files}/xtrain_{i}_split{split}.npy'
        path_y_train = f'{path_files}/ytrain_{i}_split{split}.npy'
        X_train = np.load(path_X_train)
        y_train = np.load(path_y_train)

        path_X_val = f'{path_files}/xval_{i}_split{split}.npy'
        path_y_val = f'{path_files}/yval_{i}_split{split}.npy'
        X_val = np.load(path_X_val)
        y_val = np.load(path_y_val)

        path_X_test = f'{path_files}/xtest_{i}_split{split}.npy'
        path_y_test = f'{path_files}/ytest_{i}_split{split}.npy'
        X_test = np.load(path_X_test)
        y_test = np.load(path_y_test)

        # network definition
        input_shape = (n_steps_in,n_features)
        output_shape = n_steps_out
        model = lstm_definition(input_shape,output_shape)

        #network training
        print('***Starting training with lstm***',flush=True)
        model.fit(X_train,y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=2)

        print('***Starting testing with lstm***',flush=True)
        scores = model.evaluate(X_test,y_test,verbose=2)
        print(model.metrics_names)
        print(scores)

        # evaluate complete metrics 
        forecasts = model.predict(X_test)
        path_scaler = f'{path_files}/scalers/scaler_{i}_split{split}.pkl'
        scaler = pickle.load(open(path_scaler, 'rb'))


        # results inversion
        forecasts_inverted = inverse_transform(scaler,forecasts)
        y_test_inverted = inverse_transform(scaler,y_test)
        # valutazione dei risultati
        AE, SE = evaluate_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)
        APE = evaluate_MAPE_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)

        print('Saving model and results',flush=True)
        path_model = f'{path_files}/models/local_lstm_{i}_split{split}.h5'
        model.save(path_model)

        path_forecasts = f'{path_files}/results/forecasts_local_lstm_{i}_split{split}.npy'
        np.save(path_forecasts,forecasts)

        path_AE = f'{path_files}/results/AE_local_lstm_{i}_split{split}.pkl'
        AE.to_pickle(path_AE)

        path_SE = f'{path_files}/results/SE_local_lstm_{i}_split{split}.pkl'
        SE.to_pickle(path_SE)
        
        path_APE = f'{path_files}/results/APE_local_lstm_{i}_split{split}.pkl'
        APE.to_pickle(path_APE)