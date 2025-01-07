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
epochs = 3
num_clients = 5
num_rounds = 25
splits = range(5)

path_files = '../../data/input_data'

def weight_scaling_factor(X_train,i):
    # Get the total number of training data points across all clients
    global_count = sum(X.shape[0] for X in X_train)

    # Calculate the weight scaling factor for each client
    scaling_factor = X_train[i].shape[0] / global_count

    return scaling_factor


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad



print(tf.config.list_physical_devices('GPU'),flush=True)

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


    #opening average stops values for clients: 
    path_scaler = f'{path_files}/scalers/scaler_split{split}.pkl'
    with open(path_scaler, 'rb') as f:
        scaler = pickle.load(f)

     # network definition
    print('Defining global model for fedAvg training', flush=True)
    input_shape = (n_steps_in,n_features)
    output_shape = n_steps_out
    global_model = lstm_definition(input_shape,output_shape)
    print('\n',flush = True)

    accuracy_df = pd.DataFrame(columns=['test_accuracy', 'validation_accuracy', 'trainin_accuracy'])

    for r in range(num_rounds):
        print(f'*********starting round {r+1} of {num_rounds}*********',flush=True)
        global_weights = global_model.get_weights() #getting global weights
        scaled_local_weight_list = list()
        
        for i in range(len(X_train)):
            print(f'training client {i} for round {r+1}',flush=True)
            local_model = lstm_definition(input_shape, output_shape) 
            local_model.set_weights(global_weights) # initialize with weights of the k-th rount
            local_model.fit(X_train[i],y_train[i], epochs=epochs, verbose=2, validation_data=(X_val[i], y_val[i]),batch_size = 32)
            
            #scale the model weights and add to list
            scaling_factor = weight_scaling_factor(X_train, i)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            
            #clear session to free memory after each communication round
            K.clear_session()
        
        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)
        #update global model 
        global_model.set_weights(average_weights)
        #evaluate global model
        training_scores = global_model.evaluate(X_train_global, y_train_global, verbose=0)
        validation_scores = global_model.evaluate(X_val_global, y_val_global, verbose=0)
        test_scores = global_model.evaluate(X_test_global, y_test_global, verbose=0)

        accuracy_df = accuracy_df.append({'test_accuracy': test_scores[1], 'validation_accuracy': validation_scores[1], 
                                        'training_accuracy': training_scores[1]}, ignore_index=True)

        print(f'Round {r+1} ended. Training accuracy: {training_scores[1]}, Validation accuracy: {validation_scores[1]}, Test accuracy: {test_scores[1]}')
        print('\n')


    print('***********final testing of global model*********************',flush=True)
    scores = global_model.evaluate(X_test_global,y_test_global,verbose=2)
    print(global_model.metrics_names,flush=True)
    print(scores,flush=True)

    # evaluate complete metrics 
    forecasts = global_model.predict(X_test_global)


    # results inversion
    forecasts_inverted = inverse_transform(scaler,forecasts)
    y_test_inverted = inverse_transform(scaler,y_test_global)
    # full results evaluation
    AE, SE = evaluate_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)
    APE = evaluate_MAPE_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)

    print('Saving model and results',flush=True)
    path_model = f'{path_files}/models/fedavg_lstm_split{split}.h5'
    global_model.save(path_model)

    path_accuracy_df = f'{path_files}/models/history_fedavg_lstm_split{split}.pkl'
    accuracy_df.to_pickle(path_accuracy_df)

    #path_forecasts = f'{path_files}/results/forecasts_fedavg_lstm_split{split}.npy'
    #np.save(path_forecasts,forecasts)

    path_AE = f'{path_files}/results/AE_fedavg_lstm_split{split}.pkl'
    AE.to_pickle(path_AE)

    path_SE = f'{path_files}/results/SE_fedavg_lstm_split{split}.pkl'
    SE.to_pickle(path_SE)
    
    path_APE = f'{path_files}/results/APE_fedavg_lstm_split{split}.pkl'
    APE.to_pickle(path_APE)
    
    
    print(f'****Starting testing on local datasets****')
    for i in range(num_clients):
        print(f'***client {i} ***')
        path_model = f'{path_files}/models/fedavg_lstm_split{split}.h5'
        local_federated_model = tf.keras.models.load_model(path_model)
        scores = local_federated_model.evaluate(X_test[i],y_test[i],verbose=2)
        print(local_federated_model.metrics_names)
        print(scores)
        # evaluate complete metrics 
        forecasts = local_federated_model.predict(X_test[i])
        # results inversion
        forecasts_inverted = inverse_transform(scaler,forecasts)
        y_test_inverted = inverse_transform(scaler,y_test[i])
        # valutazione dei risultati
        AE, SE = evaluate_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)
        APE = evaluate_MAPE_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)
        print('Saving results',flush=True)
        #path_forecasts = f'{path_files}/results/forecasts_local_custom3_federated_lstm_city_client_{areas[i]}_split{split}.npy'
        #np.save(path_forecasts,forecasts)
        path_AE = f'{path_files}/results/AE_local_fedavg_lstm_{i}_split{split}.pkl'
        AE.to_pickle(path_AE)
        path_SE = f'{path_files}/results/SE_local_fedavg_lstm_{i}_split{split}.pkl'
        SE.to_pickle(path_SE)
        path_APE = f'{path_files}/results/APE_local_fedavg_lstm_{i}_split{split}.pkl'
        APE.to_pickle(path_APE)

