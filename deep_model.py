import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Conv1D, GRU, Bidirectional
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import Huber, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import matplotlib.pyplot as plt

class ModelDeep:
    def autoencoder(data):
        #clear past Learnings 
        tf.keras.backend.clear_session()

        #build our model
        input = Input(shape=(data.shape[1],))
        encoded = Dense(128, activation="relu")(input)
        encoded = Dense(64, activation="relu")(encoded)
        encoded = Dense(32, activation="relu")(encoded)
        encoded = Dense(2, activation="relu")(encoded)

        decoded = Dense(32, activation="relu")(encoded)
        decoded = Dense(64, activation="relu")(decoded)
        decoded = Dense(128, activation="relu")(decoded)
        decoded = Dense(data.shape[1])(decoded)
        autoencoder = Model(input, decoded)
        
        return autoencoder

    def lstm(data):
        #clear past Learnings 
        tf.keras.backend.clear_session()

        #build our model
        lstm = Sequential([
        LSTM(128, return_sequences=True, input_shape=[data.shape[1], data.shape[2]]),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid") 
        ])

        return lstm

    def lstm_conv(data):
        #clear past Learnings 
        tf.keras.backend.clear_session()

        #build our model        
        lsco = Sequential([
        Conv1D(filters=256, kernel_size=3,
                            strides=1,
                            activation="relu",
                            padding='causal',
                            input_shape=[data.shape[1], data.shape[2]]),
                            
        LSTM(128, return_sequences=True),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(1)
        ])

        return lsco

    def bidirectional_lstm(data):
        #clear past Learnings 
        tf.keras.backend.clear_session()

        #build our model  
        bi_lstm = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=[data.shape[1], data.shape[2]]),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128)),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(1) 
        ])

        return bi_lstm

    def bidirectional_lstm_conv(data):
        #clear past Learnings 
        tf.keras.backend.clear_session()

        #build our model  
        bi_lsco = Sequential([
        Conv1D(filters=256, kernel_size=3,
                            strides=1,
                            activation="relu",
                            padding='causal',
                            input_shape=[data.shape[1], data.shape[2]]),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128)),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(1) 
        ])

        return bi_lsco

    def gru(data):
        #clear past Learnings 
        tf.keras.backend.clear_session()

        #build our model  
        gru = Sequential([
        GRU(128, return_sequences=True, input_shape=[data.shape[1], data.shape[2]]),
        GRU(128, return_sequences=True),
        GRU(128),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(1) 
        ])

        return gru

    def gru_conv(data):
        #clear past Learnings 
        tf.keras.backend.clear_session()

        #build our model  
        grco = Sequential([
        Conv1D(filters=256, kernel_size=3,
                            strides=1,
                            activation="relu",
                            padding='causal',
                            input_shape=[data.shape[1], data.shape[2]]),
        GRU(128, return_sequences=True),
        GRU(128, return_sequences=True),
        GRU(128),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(1) 
        ])

        return grco

    def gru_lstm(data):
        #clear past Learnings 
        tf.keras.backend.clear_session()

        #build our model  
        gru_lstm = Sequential([
        LSTM(256, return_sequences=True, input_shape=[data.shape[1], data.shape[2]]),
        LSTM(256, return_sequences=True),        
        GRU(256, return_sequences=True),
        GRU(256, return_sequences=True),
        GRU(256),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(1) 
        ])

        return gru_lstm

