import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

plt.imshow(X_train[5])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

import tensorflow.keras.utils as utl
y_train = utl.to_categorical(y_train,10)
y_test = utl.to_categorical(y_test,10)
X_train /= 255
X_test /= 255
inputShape = X_train.shape[1:]
# #######################
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto() 
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)
# ###########################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten

classifier = Sequential()
classifier.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu',input_shape=inputShape))
classifier.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu'))
classifier.add(MaxPooling2D((2,2)))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu'))
classifier.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu'))
classifier.add(MaxPooling2D((2,2)))
classifier.add(Dropout(0.2))

classifier.add(Flatten())

classifier.add(Dense(units = 512 , activation='relu'))
classifier.add(Dense(units = 512 , activation='relu'))

classifier.add(Dense(units = 10 , activation='softmax'))

classifier.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

hist = classifier.fit(X_train,y_train ,batch_size = 25 , epochs = 10)

evaluation = classifier.evaluate(X_test,y_test)

y_pred = classifier.predict(X_test)
pred = classifier.predict_classes(X_test)
true = y_test.argmax(1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true, pred)

import os
dirc = os.path.join(os.getcwd(),'models')

if not os.path.isdir(dirc):
    os.makedirs(dirc)
model_path = os.path.join(dirc,'cifar10_trained_model.h5')
classifier.save(model_path)