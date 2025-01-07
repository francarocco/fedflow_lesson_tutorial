import tensorflow as tf


# LSTM network definition
def lstm_definition(input_shape, output_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=input_shape,return_sequences=True)) # -> 1 input layer, 1  intermediate layer
    model.add(tf.keras.layers.LSTM(128,return_sequences=True)) # -> another hidden 
    model.add(tf.keras.layers.LSTM(32)) # -> another hidden 
    model.add(tf.keras.layers.Dense(output_shape))  # -> output layer
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(),metrics=['mean_absolute_error','mean_absolute_percentage_error','mean_squared_error'])
    return model

# CNN network defintion
def cnn_definition(input_shape, output_shape):
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
