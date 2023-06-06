# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:52:36 2023

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



new_model = tf.keras.models.load_model('saved_model/Autoencoder_AnomalyECG2')

# Check its architecture
new_model.summary()

dataframe = pd.read_csv("C:/Users/zebaa/.spyder-py3/ecg.csv")
raw_data = dataframe.values
dataframe.head()

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=21)

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

test_data = (test_data - min_val) / (max_val - min_val)
test_data = tf.cast(test_data, tf.float32)
test_labels = test_labels.astype(bool)
normal_test_data = test_data[test_labels]
anomaly_test_data = test_data[~test_labels]

plt.plot(normal_test_data[0])
plt.plot(normal_test_data[1])
plt.plot(normal_test_data[2])
plt.title("Normal data")

plt.plot(anomaly_test_data[0])
plt.plot(anomaly_test_data[1])
plt.plot(anomaly_test_data[2])
plt.title("Anomaly data")


reconstruction_n = new_model.predict(normal_test_data)
reconstruction_a = new_model.predict(anomaly_test_data)


plt.plot(normal_test_data[4],'b')
plt.plot(reconstruction_n[4],'r')
plt.fill_between(np.arange(140),reconstruction_n[4],normal_test_data[4],color='lightcoral')

out = mean(abs(np.subtract(normal_test_data[10],reconstruction_n[10])))
out

plt.plot(anomaly_test_data[4],'b')
plt.plot(reconstruction_a[4],'r')
plt.fill_between(np.arange(140),reconstruction_a[4],anomaly_test_data[4],color='lightcoral')

out = mean(abs(np.subtract(anomaly_test_data[4],reconstruction_a[4])))
out
    
n = 5
plt.figure(figsize=(20, 5))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.plot(normal_test_data[i],'b')
  plt.plot(reconstruction_n[i],'r')
  plt.fill_between(np.arange(140),reconstruction_n[i],normal_test_data[i],color='lightcoral')
  plt.title("Model performance on Normal data")

plt.show()

n = 5
plt.figure(figsize=(20, 5))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.plot(anomaly_test_data[i],'b')
  plt.plot(reconstruction_a[i],'r')
  plt.fill_between(np.arange(140),reconstruction_a[i],anomaly_test_data[i],color='lightcoral')
  plt.title("Model performance on Anomaly data")

plt.show()

train_loss_n = tf.keras.losses.mae(reconstruction_n, normal_test_data)
plt.hist(train_loss_n, bins=50)
plt.title("loss on normal test data")

train_loss_n[10]

train_loss_a = tf.keras.losses.mae(reconstruction_a, anomaly_test_data)
plt.hist(train_loss_a, bins=50)
plt.title("loss on anomaly test data")

threshold = np.mean(train_loss_n) + 2*np.std(train_loss_n)

#np.mean(train_loss_n)
#np.std(train_loss_n)
#2*np.std(train_loss_n)
#np.mean(train_loss_n) + 2*np.std(train_loss_n)

plt.hist(train_loss_n, bins=50, label='normal')
plt.hist(train_loss_a, bins=50, label='anomaly')
plt.axvline(threshold, color='r', linewidth=3, linestyle='dashed', label='{:0.3f}'.format(threshold))
plt.legend(loc='upper right')
plt.title("Normal and Anomaly Loss")
plt.show()

########## false positive
count_fp = 0
for i in range(len(train_loss_a)):
    if(train_loss_a[i]<threshold):
        count_fp = count_fp + 1
        print('id',i,'error',train_loss_a[i])
        print('verification',mean(abs(np.subtract(anomaly_test_data[i],reconstruction_a[i]))))
        

fp = round((count_fp/len(train_loss_a))*100,1)

print('false positive percentage - ',fp,' , false positive instances - ',
      count_fp,' , total anomoulous data-',len(train_loss_a))


########## false negative    
count_fn = 0
for i in train_loss_n:
    if(i>threshold):
        count_fn = count_fn + 1
fn = round((count_fn/len(train_loss_n))*100,1)

print('false negative percentage - ',fn,' , false negative instances - ',
      count_fn,' , total normal data-',len(train_loss_n))


########### true positive  
count_tp = 0
for i in train_loss_n:
    if(i<threshold):
        count_tp = count_tp + 1
tp = round((count_tp/len(train_loss_n))*100,1)
 
print('true positive percentage - ',tp,' , true positive instances - ',
      count_tp,' , total normal data-',len(train_loss_n))

################### true negative
count_tn = 0
for i in train_loss_a:
    if(i>threshold):
        count_tn = count_tn + 1
tn = round((count_tn/len(train_loss_a))*100,1)

print('true negative percentage - ',tn,' , true negative instances - ',
      count_tn,' , total anomoulous data-',len(train_loss_a))



################   PERFORMANCE

#count_tp+count_tn+count_fp+count_fn
#len(train_loss_a)+len(train_loss_n)

print('accuracy-',round(((count_tp+count_tn)/(count_tp+count_tn+count_fp+count_fn)),2))
print('precision-',round(((count_tp)/(count_tp+count_fp)),2))
print('recall-',round(((count_tp)/(count_tp+count_fn)),2))
print('Specificity-',round(((count_tn)/(count_tn+count_fp)),2))

############ anomaly checking

import random

reconstruction = new_model.predict(test_data)

i = random.randint(0,len(test_data))
print(i)

loss = tf.keras.losses.mae(reconstruction[i], test_data[i])
if(loss>threshold):
    print('Anomaly')
else:
    print('Normal')
    
plt.plot(test_data[i],'b')
plt.plot(reconstruction[i],'r')
plt.fill_between(np.arange(140),reconstruction[i],test_data[i],color='lightcoral')
out = mean(abs(np.subtract(test_data[i],reconstruction[i])))
print(out)


