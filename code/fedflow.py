# importing libraries
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

import sys
import os


from utilities.utilities import *
from utilities.networks import * 

class FedFlow:

    def __init__(self):  
        self = self

      
    def calculate_scaling_factor(self,similarity_lists,i): # calculate scaling factors for client i
        return similarity_lists[i]

    def weight_scaling_factor(self,X_train,i):
        # Get the total number of training data points across all clients
        global_count = sum(X.shape[0] for X in X_train)

        # Calculate the weight scaling factor for each client
        scaling_factor = X_train[i].shape[0] / global_count

        return scaling_factor

    def scale_model_weights(self,weight, scalar):
        '''function for scaling a models weights'''
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final



    def sum_scaled_weights(self,scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
            
        return avg_grad
        
    def custom_scaled_weights_client(self,local_weight_list, client_scaling_factor): # calculate new weights for client i
        num_clients = len(local_weight_list)
        print(f'len local weight list: {len(local_weight_list)}', flush=True)
        print(f'client scaling factor: {len(client_scaling_factor)}', flush=True)
        temp_weight_i = []
        for j in range(num_clients):
            weight_j = self.scale_model_weights(local_weight_list[j],client_scaling_factor[j])
            temp_weight_i.append(weight_j)
        updated_local_weigths = self.sum_scaled_weights(temp_weight_i)
        return updated_local_weigths

    def calculate_similarities(self,context_vectors): # calculate similarities among context vectors
        # normalize clients context vectors
        context_vectors[:-1,:] = min_max_normalize(context_vectors[:-1,:])
        context_vectors[-1:,:] = min_max_normalize(context_vectors[-1:,:])

        similarity_lists = [] # calculate similarities based on Euclidean Distances
        euclidean_distances = np.zeros((context_vectors.shape[1], context_vectors.shape[1]))
        for i in range(context_vectors.shape[1]):
            for j in range(context_vectors.shape[1]):
                distance = np.linalg.norm(context_vectors[:,i] - context_vectors[:,j])
                euclidean_distances[i,j] = distance
        for i in range(euclidean_distances.shape[0]):
            euclidean_distances_area = euclidean_distances[i,:]
            euclidean_distances_area = 1/(1+euclidean_distances_area)
            euclidean_distances_area = euclidean_distances_area/sum(euclidean_distances_area)
            similarity_lists.append(euclidean_distances_area)
        return similarity_lists

    def fedflow_models_generation(self,path_files,input_shape,output_shape,similarity_lists,X_train,y_train,X_val,y_val,split,epochs,num_rounds): # generate personalized models through FedFlow
        # network definition
        global_model = lstm_definition(input_shape,output_shape)
    
        custom_global_weights = []
        for i in range(len(X_train)):
            custom_global_weights.append(global_model.get_weights()) #randomly initialize clients' models

        for r in range(num_rounds):
            global_weights = global_model.get_weights() #getting global weights
            scaled_local_weight_list = list()
            local_weight_list = list()
            
            for i in range(len(X_train)):
                #local training for round r
                local_model = lstm_definition(input_shape, output_shape)
                local_model.set_weights(custom_global_weights[i]) #getting weights of client i
                local_model.fit(X_train[i],y_train[i], epochs=epochs, verbose=2, validation_data=(X_val[i], y_val[i]),batch_size = 32)
                
                #scale the model weights and add to list
                scaling_factor = self.weight_scaling_factor(X_train, i)
                scaled_weights = self.scale_model_weights(local_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                local_weight_list.append(local_model.get_weights())
                
                #clear session to free memory after each communication round
                K.clear_session()
            
            updated_custom_global_weights = []
            
            for i in range(len(X_train)):
                client_scaling_factor = self.calculate_scaling_factor(similarity_lists,i) #taking the weights for client i, based on the similarities
                updated_local_weigths = self.custom_scaled_weights_client(local_weight_list, client_scaling_factor)
                updated_custom_global_weights.append(updated_local_weigths)

            

            #calculate local model based on custom strategy
            custom_global_weights = updated_custom_global_weights.copy()
            
    

        for i in range(len(X_train)):
            # Saving local models
            local_model = lstm_definition(input_shape, output_shape)
            local_model.set_weights(custom_global_weights[i])
            path_local_model = f'{path_files}/models/fedflow_city_client{i}_split{split}.h5'
            local_model.save(path_local_model)


    def fedflow_finetuning(self,path_files,X_train,y_train,X_val,y_val,X_test,y_test,split,epochs_finetuning,scaler,n_steps_in,n_steps_out):   
        for i in range(len(X_train)):
                    
            # working with local model obtained through FedFlow
            path_model = f'{path_files}/models/fedflow_city_client{i}_split{split}.h5'
            local_federated_model = tf.keras.models.load_model(path_model)
            
            #Starting fine tuning of client i
            print(f'***Starting fine-tuning for client{i}***',flush=True)
            local_federated_model.fit(X_train[i],y_train[i], epochs=epochs_finetuning, verbose=2, validation_data=(X_val[i], y_val[i]),batch_size = 32)
            print('***Starting testing***',flush=True)
            scores = local_federated_model.evaluate(X_test[i],y_test[i],verbose=2)
            
            # evaluate complete metrics 
            forecasts = local_federated_model.predict(X_test[i])
            # results inversion
            forecasts_inverted = inverse_transform(scaler,forecasts)
            y_test_inverted = inverse_transform(scaler,y_test[i])
            # valutazione dei risultati
            AE, SE = evaluate_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)
            APE = evaluate_MAPE_forecasts(y_test_inverted, forecasts_inverted, n_steps_in, n_steps_out)

            print('Saving model and results',flush=True)
            path_model = f'{path_files}/models/local_fedflow_city_client{i}_split{split}.h5'
            local_federated_model.save(path_model)

            path_forecasts = f'{path_files}/results/forecasts_local_fedflow_city_client{i}_split{split}.npy'
            np.save(path_forecasts,forecasts)

            path_AE = f'{path_files}/results/AE_local_fedflow_city_client{i}_split{split}.pkl'
            AE.to_pickle(path_AE)

            path_SE = f'{path_files}/results/SE_fedflow_city_client{i}_split{split}.pkl'
            SE.to_pickle(path_SE)
            
            path_APE = f'{path_files}/results/APE_fedflow_city_client{i}_split{split}.pkl'
            APE.to_pickle(path_APE)
