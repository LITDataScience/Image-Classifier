import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

"""
   STILL UNDER DEVELOPMENT

"""
dataset = pd.read_csv("G:\\AI\\Deep-MNIST-image-Classifier\\data\\Salary_Data.csv")

x = dataset.values.reshape(-1,1).astype(np.float64)
y = dataset.values.reshape(-1,1).astype(np.float64)

"""
Set up the logistic regression model
"""
sc = MinMaxScaler(feature_range=(-1,1))
x_ = sc.fit_transform(x)
y_ = sc.fit_transform(y)

model = Sequential()
model.add(Dense(2,  # output dim is 2, one score per each class
                activation='softmax',
                kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                input_dim=len(feature_scale)))  # input dimension = number of features your data has
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
model.fit(x_, y_, epochs=100, validation_data=(model.predict(x_), model.predict(y_)),validation_split=1/3, callbacks=[early_stopping_monitor])