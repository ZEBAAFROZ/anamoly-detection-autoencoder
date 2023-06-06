# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:33:57 2023

@author: zebaa
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statistics import mean

dataframe = pd.read_csv("C:/Users/zebaa/.spyder-py3/ecg.csv")
raw_data = dataframe.values
dataframe.head()

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomaly_train_data = train_data[~train_labels]
anomaly_test_data = test_data[~test_labels]

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
                           tf.keras.layers.Dense(64, activation='relu'),
                           tf.keras.layers.Dense(32, activation='relu'),
                           tf.keras.layers.Dense(16, activation='relu'),
                           tf.keras.layers.Dense(8, activation='relu')
                         ])
        self.decoder = tf.keras.Sequential([
                           tf.keras.layers.Dense(16, activation='relu'),
                           tf.keras.layers.Dense(32, activation='relu'),
                           tf.keras.layers.Dense(64, activation='relu'),
                           tf.keras.layers.Dense(140, activation='sigmoid')
                         ])
            
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = Autoencoder()


earlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2, mode = 'min')
model.compile(optimizer = 'adam',  loss = 'mae')

history = model.fit(normal_train_data, normal_train_data, epochs = 50, batch_size = 120,
                    validation_data = (normal_test_data,normal_test_data),
                    shuffle = True,
                    callbacks = [earlyStopping])

model.summary()
#model.save('Autoencoder_AnomalyECG2.h5')

model.save('saved_model/Autoencoder_AnomalyECG2')

e = model.encoder(normal_test_data).numpy()
d = model.decoder(e).numpy()

plt.plot(normal_test_data[5],'b')
plt.plot(d[5],'r')
plt.fill_between(np.arange(140),d[5],normal_test_data[5],color='lightcoral')

normal_test_data[0]
d[0]

reconstruction = model.predict(np.expand_dims(normal_test_data[0]), axis=0)
loss_0 = tf.keras.losses.mae(reconstruction,normal_test_data[0])

out = mean(abs(np.subtract(normal_test_data[5],d[5])))
out

normal_train_data[0].shape()

reconstruction = model.predict(normal_test_data)

plt.plot(normal_test_data[6],'b')
plt.plot(reconstruction[6],'r')
plt.fill_between(np.arange(140),reconstruction[6],normal_test_data[6],color='lightcoral')

out = mean(abs(np.subtract(normal_test_data[6],reconstruction[6])))
out
