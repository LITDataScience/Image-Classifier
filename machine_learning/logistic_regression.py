import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

dataset = pd.read_csv('/home/starlord/Desktop/DEEP-MNIST-IMAGE-CLASSIFIER/Deep-MNIST-image-Classifier/data/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  ## Columns to be feature scaled.
y = dataset.iloc[:, 4].values ## Column for Dependant variable.

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import  StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()
model.add(Dense(16, input_shape=(3,)))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score, accuracy = model.evaluate(y_test, model.predict(y_test), batch_size=16, verbose=0)
print("Test fraction correct (NN-Score) = {:.2f}".format(score))
print("Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))