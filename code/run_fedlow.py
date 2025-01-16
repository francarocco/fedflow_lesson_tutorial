# importing libraries
import pickle
import numpy as np

import tensorflow as tf


from tensorflow.keras import backend as K

from utilities.utilities import *
from utilities.networks import *

from fedflow import FedFlow


n_steps_in = 9      # hi: history length -> number of samples used for prediction
n_steps_out = 3     # ho: horizon -> number of future stops to predict
n_features = 6      # number of considered features (occupancy, stop_id, month, weekday, timeslot, route_id)
epochs = 3          # E: number of local epochs 
num_clients = 5     # N: number of considered clients (i.e. transport companies)
num_rounds = 25     # K: number of federated learning rounds
epochs_finetuning = 3   # Ef: number of epochs used for fine-tuning 
splits = range(5)

print(tf.config.list_physical_devices('GPU'),flush=True)

# the federated learning process is distributed, but because we are interested in evaluating accuracy, 
# to facilitate validation of the framework we ran the process sequentially on a single node, 
# where the server and clients run

path_files = '../data/input_data'


for split in splits:

    print(f'Reading clients datasets for split {split}', flush=True)
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    for i in range(num_clients):
        path_X_test_i = f'{path_files}/xtest_{i}_split{split}.npy'
        path_y_test_i = f'{path_files}/ytest_{i}_split{split}.npy'
        X_test_i = np.load(path_X_test_i)
        y_test_i = np.load(path_y_test_i)
        path_X_val_i = f'{path_files}/xval_{i}_split{split}.npy'
        path_y_val_i = f'{path_files}/yval_{i}_split{split}.npy'
        X_val_i = np.load(path_X_val_i)
        y_val_i = np.load(path_y_val_i)
        path_X_train_i = f'{path_files}/xtrain_{i}_split{split}.npy'
        path_y_train_i = f'{path_files}/ytrain_{i}_split{split}.npy'
        X_train_i = np.load(path_X_train_i)
        y_train_i = np.load(path_y_train_i)
        
        X_train.append(X_train_i)
        y_train.append(y_train_i)
        X_val.append(X_val_i)
        y_val.append(y_val_i)
        X_test.append(X_test_i)
        y_test.append(y_test_i)


    # dataset have been already generated for LSTM
    print('Reading global training, validation, test set',flush=True)
    path_X_train_global = f'{path_files}/xtrain_city_split{split}.npy'
    path_y_train_global = f'{path_files}/ytrain_city_split{split}.npy'
    X_train_global = np.load(path_X_train_global)
    y_train_global = np.load(path_y_train_global)
    path_X_val_global = f'{path_files}/xval_city_split{split}.npy'
    path_y_val_global = f'{path_files}/yval_city_split{split}.npy'
    X_val_global = np.load(path_X_val_global)
    y_val_global = np.load(path_y_val_global)
    path_X_test_global = f'{path_files}/xtest_city_split{split}.npy'
    path_y_test_global = f'{path_files}/ytest_city_split{split}.npy'
    X_test_global = np.load(path_X_test_global)
    y_test_global = np.load(path_y_test_global)

    

    # list of N context vectors, one for each client
    path_factors = f'{path_files}/context_vectors.pkl'
    with open(path_factors, 'rb') as f:
        context_vectors = pickle.load(f)

    # scaler 
    path_scaler = f'{path_files}/scalers/scaler_split{split}.pkl'
    with open(path_scaler, 'rb') as f:
        scaler = pickle.load(f)

    ff = FedFlow()

    input_shape = (n_steps_in,n_features)
    output_shape = n_steps_out

    similarity_lists = ff.calculate_similarities(context_vectors) # at round 0 the server calculates the similarities among clients
    ff.fedflow_models_generation(path_files,input_shape,output_shape,similarity_lists,X_train,y_train,X_val,y_val,split,epochs) # perform local models generation with FedFlow
    ff.fedflow_finetuning(path_files,X_train,y_train,X_val,y_val,X_test,y_test,split,epochs_finetuning,scaler,n_steps_in,n_steps_out) # final fine-tuning for further personalization and testing
    