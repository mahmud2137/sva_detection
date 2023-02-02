import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys
sys.modules['keras'] = keras
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, LSTM, Flatten, Dense, Input, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data_prep import *
import os



class AutoEncoder_eda():
    def __init__(self, input_shape, bottle_neck_shape = 20):
        self.input_shape = input_shape
        self.bottle_neck_shape = bottle_neck_shape
        self.build_model()

    def encoder_layers(self,x):
        x = LSTM(100, return_sequences = True)(x)
        x = LSTM(50, return_sequences = False)(x)
        x = Flatten()(x)
        # x = Dense(50, activation='relu')(x)
        encoded = Dense(self.bottle_neck_shape, activation='relu')(x)
        return encoded

    def decoder_layers(self, x):
        
        x = Dense(5*50, activation='relu')(x)
        x = Reshape((5, 50))(x)
        x = LSTM(50, return_sequences = True)(x)
        x = LSTM(100, return_sequences = True)(x)
        decoded = LSTM(2, return_sequences=True)(x)
        return decoded
    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        self.bottleneck_layer = self.encoder_layers(input_layer)
        output_layer = self.decoder_layers(self.bottleneck_layer)
        self.encoder = Model(input_layer, self.bottleneck_layer)
        self.autoencoder = Model(input_layer, output_layer)

class AutoEncoder_eeg():
    def __init__(self, input_shape, bottle_neck_shape = 20):
        self.input_shape = input_shape
        self.bottle_neck_shape = bottle_neck_shape
        self.build_model()

    def encoder_layers(self,x):
        x = LSTM(100, return_sequences = True)(x)
        x = LSTM(50, return_sequences = False)(x)
        x = Flatten()(x)
        # x = Dense(50, activation='relu')(x)
        encoded = Dense(self.bottle_neck_shape, activation='relu')(x)
        return encoded

    def decoder_layers(self, x):
        
        x = Dense(10*50, activation='relu')(x)
        x = Reshape((10, 50))(x)
        x = LSTM(50, return_sequences = True)(x)
        x = LSTM(100, return_sequences = True)(x)
        decoded = LSTM(5, return_sequences=True)(x)
        return decoded
    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        self.bottleneck_layer = self.encoder_layers(input_layer)
        output_layer = self.decoder_layers(self.bottleneck_layer)
        self.encoder = Model(input_layer, self.bottleneck_layer)
        self.autoencoder = Model(input_layer, output_layer)


if __name__ == '__main__':
    eda_label = pd.read_csv("data/eda_labels.csv")
    x, y = eda_preprop(eda_label)

    eeg_label = pd.read_csv("data/eeg_labels.csv")
    x_, y_ = eeg_preprep(eeg_label)
    
    aeeeg = AutoEncoder_eeg(x_.shape[1:])
    aeeeg.autoencoder.summary()


    aeeda = AutoEncoder_eda(x.shape[1:])
    

    input_layer = Input(shape=x.shape[1:])
    lstm_model = Model(input_layer, model_layers(input_layer))  
    opt = Adam(lr=0.0008)
    lstm_model.compile(loss = 'mse', optimizer = opt)
    lstm_model.summary()
    hist = lstm_model.fit(x,y,
                    epochs=10,
                    batch_size=128,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x, y))
    
