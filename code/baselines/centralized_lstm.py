
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
    path_scaler = f'{path_files}/scalers/scaler_split{split}.pkl'
    scaler = pickle.load(open(path_scaler, 'rb'))

    print(f'******Opening data for split {split}******', flush = True)
    # opening datasets organized by clients
    # dataset have been already generated for lstm
    print('Reading training, validation, test set',flush=True)
    path_X_train = f'{path_files}/xtrain_city_split{split}.npy'
    path_y_train = f'{path_files}/ytrain_city_split{split}.npy'
    X_train = np.load(path_X_train)
    y_train = np.load(path_y_train)

    path_X_val = f'{path_files}/xval_city_split{split}.npy'
    path_y_val = f'{path_files}/yval_city_split{split}.npy'
    X_val = np.load(path_X_val)
    y_val = np.load(path_y_val)
        
    path_X_test = f'{path_files}/xtest_city_split{split}.npy'
    path_y_test = f'{path_files}/ytest_city_split{split}.npy'
    X_test = np.load(path_X_test)
    y_test = np.load(path_y_test)

    # network definition
    input_shape = (n_steps_in,n_features)
    output_shape = n_steps_out
    model = lstm_definition(input_shape,output_shape)

    #network training
    print('***Starting training with lstm***',flush=True)
    model.fit(X_train,y_train, validation_data=(X_val, y_val), epochs=100, verbose=2)

    print('***Starting testing with lstm***',flush=True)
    scores = model.evaluate(X_test,y_test,verbose=2)

    # evaluate complete metrics 
    forecasts = model.predict(X_test)


    # results inversion
    forecasts_inverted = inverse_transform(scaler,forecasts)
    y_test_inverted = inverse_transform(scaler,y_test)
    # results evaluation
    AE, SE = evaluate_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)
    APE = evaluate_MAPE_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)
    
    print('Saving model and results',flush=True)
    path_model = f'{path_files}/models/centralized_lstm_split{split}.h5'
    model.save(path_model)

    path_forecasts = f'{path_files}/results/forecasts_centralized_lstm_split{split}.npy'
    np.save(path_forecasts,forecasts)

    path_AE = f'{path_files}/results/AE_centralized_lstm_split{split}.pkl'
    AE.to_pickle(path_AE)

    path_SE = f'{path_files}/results/SE_centralized_lstm_split{split}.pkl'
    SE.to_pickle(path_SE)
    
    path_APE = f'{path_files}/results/APE_centralized_lstm_split{split}.pkl'
    APE.to_pickle(path_AE)
