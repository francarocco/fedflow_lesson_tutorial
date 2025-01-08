import tensorflow as tf


# LSTM network definition
def lstm_definition(input_shape, output_shape):
    """
    Defines a 3-layer Long Short-Term Memory (LSTM) neural network for time series prediction.

    Parameters:
    ----------
    input_shape : tuple
        The shape of the input data. It includes the time steps and the number of features (num_timesteps,num_features)
    output_shape : int
        The horizon of the prediction.

    Returns:
    -------
    model : tf.keras.models.Sequential
        A compiled LSTM-based neural network model.

    Model Architecture:
    -------------------
    1. Input Layer: LSTM layer with 128 units, configured to return sequences.
    2. Hidden Layer 1: Another LSTM layer with 128 units, configured to return sequences.
    3. Hidden Layer 2: LSTM layer with 32 units (no sequence return).
    4. Output Layer: Fully connected (Dense) layer with `output_shape` neurons for predictions.

    Loss Function:
    --------------
    The model is compiled with 'mean_absolute_error' as the loss function and uses Adam optimizer.
    Metrics:
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    - Mean Squared Error (MSE)
    """
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=input_shape,return_sequences=True)) # -> 1 input layer, 1  intermediate layer
    model.add(tf.keras.layers.LSTM(128,return_sequences=True)) # -> another hidden 
    model.add(tf.keras.layers.LSTM(32)) # -> another hidden 
    model.add(tf.keras.layers.Dense(output_shape))  # -> output layer
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(),metrics=['mean_absolute_error','mean_absolute_percentage_error','mean_squared_error'])
    return model

# CNN network defintion
def cnn_definition(input_shape, output_shape):
    """
    Defines a Convolutional Neural Network (CNN) for time series prediction.

    Parameters:
    ----------
    input_shape : tuple
        The shape of the input data. It includes the time steps and the number of features (num_timesteps,num_features)
    output_shape : int
        The horizon of the prediction.

    Returns:
    -------
    model : tf.keras.models.Sequential
        A compiled CNN-based neural network model.

    Model Architecture:
    -------------------
    1. Input Layer:
        - Conv1D layer with 128 filters, kernel size 5, ReLU activation, and 'same' padding.
        - MaxPooling1D with pool size 2 for downsampling.
    2. Hidden Layers:
        - Conv1D layer with 64 filters, kernel size 5, ReLU activation, and 'same' padding.
        - MaxPooling1D with pool size 1 for minimal downsampling.
        - Conv1D layer with 128 filters, kernel size 5, ReLU activation, and 'same' padding.
        - MaxPooling1D with pool size 1 for minimal downsampling.
    3. Flatten Layer:
        - Flattens the output of the previous layer to prepare for dense layers.
    4. Fully Connected Layers:
        - Dense layer with 32 neurons and ReLU activation.
        - Dense layer with `output_shape` neurons for final predictions.

    Loss Function:
    --------------
    The model is compiled with 'mean_absolute_error' as the loss function and uses the Adam optimizer.
    Metrics:
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    - Mean Squared Error (MSE)
    """
    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))  # Adjusted pool_size here
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(output_shape))
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(), metrics=['mean_absolute_error','mean_absolute_percentage_error','mean_squared_error'])
    return model
