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
        return similarity_lists[i] # simply return the currect row for the selected client

    def weight_scaling_factor(self,X_train,i):
        # Get the total number of training data points across all clients
        global_count = sum(X.shape[0] for X in X_train)

        # Calculate the weight scaling factor for each client
        scaling_factor = X_train[i].shape[0] / global_count

        return scaling_factor

    def scale_model_weights(self,weight, scalar):
        '''function for scaling model weights'''
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
        
    # Function to calculate new weights for client i, server side    
    def custom_scaled_weights_client(self,local_weight_list, client_scaling_factor): 
        """
        Calculates the updated weights for a client by scaling the local weights of all clients
        using a scaling factor and then summing the scaled weights.

        Parameters:
        ----------
        local_weight_list : list
            A list of local model parameters from all clients. Each element in the list represents 
            the parameter of a single client's model.
        client_scaling_factor : list
            A list of scaling factors corresponding to each client, used to adjust 
            the contribution of each client's weights.

        Returns:
        -------
        updated_local_weights : np.ndarray
            The aggregated and scaled weights for the client after applying the scaling factor 
            and summing the scaled weights.


        Example Usage:
        --------------
        updated_weights = custom_scaled_weights_client(local_weight_list, client_scaling_factor)
        """
        num_clients = len(local_weight_list)
        #print(f'len local weight list: {len(local_weight_list)}', flush=True)
        #print(f'client scaling factor: {len(client_scaling_factor)}', flush=True)
        temp_weight_i = []
        for j in range(num_clients): # iterating client by client
            weight_j = self.scale_model_weights(local_weight_list[j],client_scaling_factor[j]) # Scale the weights of client j by the corresponding scaling factor (alpha_i,j)
            temp_weight_i.append(weight_j)
        updated_local_weigths = self.sum_scaled_weights(temp_weight_i) # sum to performe the average
        return updated_local_weigths

    
     # calculate similarities among clients based on context vectors
    def calculate_similarities(self,context_vectors):
        """
        Calculates the pairwise similarities among clients based on the inverse of Euclidean distances 
        of their context vectors.

        Parameters:
        ----------
        context_vectors : np.ndarray
            An array where each element represents the context vector of a client.

        Returns:
        -------
        similarity_lists : list of np.ndarray
            A list where each entry is a normalized array of similarity scores for a client 
            with respect to all other clients.


        Notes:
        ------
        - The inverse of Euclidean distance (with an added 1 to prevent division by zero) is used 
        to compute similarity scores.
        - The resulting similarities are normalized to ensure the sum of scores for each client is 1.

        Example Usage:
        --------------
        similarity_lists = calculate_similarities(context_vectors)
        """

        # context vectors are normalized using min-max normalization. 
        context_vectors[:-1,:] = min_max_normalize(context_vectors[:-1,:]) # first columns are normalized together since they represent the OSM kinds of roads
        context_vectors[-1:,:] = min_max_normalize(context_vectors[-1:,:]) # last column contains the average number of stops

        similarity_lists = [] # calculate similarities based on Euclidean Distances
        euclidean_distances = np.zeros((context_vectors.shape[1], context_vectors.shape[1])) # initialize matrix for euclidean distances among paired clients
        for i in range(context_vectors.shape[1]): # for each client
            for j in range(context_vectors.shape[1]): # calculate the distance with all the clients
                distance = np.linalg.norm(context_vectors[:,i] - context_vectors[:,j]) # euclidean distance
                euclidean_distances[i,j] = distance

        for i in range(euclidean_distances.shape[0]): # for each client (row) 
            euclidean_distances_area = euclidean_distances[i,:]
            euclidean_distances_area = 1/(1+euclidean_distances_area) # converts distances into similarity scores using the formula: similarity = 1 / (1 + distance).
            euclidean_distances_area = euclidean_distances_area/sum(euclidean_distances_area) # normalizes similarity scores for each client to sum to 1 to preserve the magnitude of the parameters
            similarity_lists.append(euclidean_distances_area)
        return similarity_lists



    def fedflow_models_generation(self,path_files,input_shape,output_shape,similarity_lists,X_train,y_train,X_val,y_val,split,epochs,num_rounds): # generate personalized models through FedFlow
        """
        Implements the Federated Learning (FL) process according to the FedFlow approach, where personalized 
        models are provided to each client. The personalization is achieved round-by-round based on client 
        similarities, calculated leveraging the context vectors.

        Parameters:
        ----------
        path_files : str
            Path to save the personalized models for each client after the FL process.
        input_shape : tuple
            The shape of the input data to the network used for prediction. It includes the time steps and the number of features (num_timesteps,num_features)
        output_shape : int
            The horizon of the prediction.
        similarity_lists : list of np.ndarray
            A list of similarity values between clients, where each entry provides the pairwise similarities for a client.
        X_train : list of np.ndarray
            A list of training data arrays for each client.
        y_train : list of np.ndarray
            A list of training labels for each client.
        X_val : list of np.ndarray
            A list of validation data arrays for each client.
        y_val : list of np.ndarray
            A list of validation labels for each client.
        split : int
            An identifier for the data split being used.
        epochs : int
            Number of training epochs for local client training.
        num_rounds : int
            Number of communication rounds in the FL process.

        Returns:
        -------
        None
            The function saves the personalized models for each client to the specified path.


        Example Usage:
        --------------
        fedflow_models_generation(path_files, input_shape, output_shape, similarity_lists, 
                                X_train, y_train, X_val, y_val, split, epochs, num_rounds)
        """
        # network definition -> define the same network for all the clients. Weights are randomly initialized
        global_model = lstm_definition(input_shape,output_shape)
    
        custom_global_weights = [] # list of different parameters for N clients (custom_global_weights[i] are the weights of client i)
        for i in range(len(X_train)):
            custom_global_weights.append(global_model.get_weights()) # randomly initialize clients' models

        for r in range(num_rounds): # iterating round by round
            global_weights = global_model.get_weights() # getting global weights
            scaled_local_weight_list = list()
            local_weight_list = list() # list of dimension N, each element represent the local parameters of client i
            
            # CLIENT SIDE
            for i in range(len(X_train)): # iterating client by client
                # local training for round r
                local_model = lstm_definition(input_shape, output_shape) # define the network structure
                local_model.set_weights(custom_global_weights[i]) # updates the weights with the weight provided by the server for client i
                local_model.fit(X_train[i],y_train[i], epochs=epochs, verbose=2, validation_data=(X_val[i], y_val[i]),batch_size = 32) # local training on local data
                
                # scale the model weights according to fedavg for baseline and add to list
                scaling_factor = self.weight_scaling_factor(X_train, i)
                scaled_weights = self.scale_model_weights(local_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)

                # save local weights of clients i in the list, for round k
                local_weight_list.append(local_model.get_weights())
                
                #clear session to free memory after each communication round
                K.clear_session()
            

            # SERVER SIDE
            # at the end of the round, the new parameters are calculated by the server
            updated_custom_global_weights = []   # list of dimension N, each element represent the updated parameters of client i 
            
            for i in range(len(X_train)): # iterating client by client
                client_scaling_factor = self.calculate_scaling_factor(similarity_lists,i) # taking the weights for client i, based on the similarities
                updated_local_weigths = self.custom_scaled_weights_client(local_weight_list, client_scaling_factor) # calculate the new parameters for client i
                updated_custom_global_weights.append(updated_local_weigths) # append the parameters to the list of updated parameters

            

            #update the parameters
            custom_global_weights = updated_custom_global_weights.copy()
            
    

        for i in range(len(X_train)): # at the end of the FL process, save the local models, client by client
            # Saving local models
            local_model = lstm_definition(input_shape, output_shape)
            local_model.set_weights(custom_global_weights[i]) # set the final weights
            path_local_model = f'{path_files}/models/fedflow_city_client{i}_split{split}.h5'
            local_model.save(path_local_model)


    def fedflow_finetuning(self,path_files,X_train,y_train,X_val,y_val,X_test,y_test,split,epochs_finetuning,scaler,n_steps_in,n_steps_out):   
        """
        Performs local fine-tuning as an additional personalization step, further specializing 
        the local models obtained from FedFlow on the clients' local datasets. Also evaluates 
        the models' performance on the local test datasets.

        Parameters:
        ----------
        path_files : str
            Path to load and save models and results.
        X_train : list of np.ndarray
            Training data for each client.
        y_train : list of np.ndarray
            Training labels for each client.
        X_val : list of np.ndarray
            Validation data for each client.
        y_val : list of np.ndarray
            Validation labels for each client.
        X_test : list of np.ndarray
            Test data for each client.
        y_test : list of np.ndarray
            Test labels for each client.
        split : int
            Identifier for the data split being used.
        epochs_finetuning : int
            Number of epochs for fine-tuning.
        scaler : object
            Scaler object used for data normalization, required for inverse transformation of results.
        n_steps_in : int
            Number of input time steps for the model (history).
        n_steps_out : int
            Number of output time steps for the model (horizon).

        Returns:
        -------
        None
            The function saves the fine-tuned models and evaluation results for each client.

        Process:
        -------
        1. Load the local model for each client from the FedFlow process.
        2. Fine-tune the local model using the client's local training data.
        3. Test the fine-tuned model on the client's local test data.
        4. Perform evaluation:
            - Generate forecasts.
            - Inverse transform the forecasts and actual test labels for interpretation.
            - Compute evaluation metrics (AE, SE, APE).
        5. Save the fine-tuned models, forecasts, and evaluation metrics.

        Example Usage:
        --------------
        fedflow_finetuning(path_files, X_train, y_train, X_val, y_val, X_test, y_test, 
                        split, epochs_finetuning, scaler, n_steps_in, n_steps_out)
        """

        for i in range(len(X_train)): # iterating client by client
                    
            # working with local model obtained through FedFlow
            path_model = f'{path_files}/models/fedflow_city_client{i}_split{split}.h5'
            local_federated_model = tf.keras.models.load_model(path_model) # take the saved local model 
            
            # Starting fine tuning of client i -> performs additional Ef epochs on the local dataset
            print(f'***Starting fine-tuning for client{i}***',flush=True)
            local_federated_model.fit(X_train[i],y_train[i], epochs=epochs_finetuning, verbose=2, validation_data=(X_val[i], y_val[i]),batch_size = 32)
            
            # test the model
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
