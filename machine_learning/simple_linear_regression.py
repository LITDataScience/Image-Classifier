import pandas as pd
import numpy as np
import keras
from matplotlib import pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model

from sklearn.preprocessing import MinMaxScaler, Normalizer


dataset = pd.read_csv("/home/starlord/Desktop/DEEP-MNIST-IMAGE-CLASSIFIER/Deep-MNIST-image-Classifier/data/Salary_Data.csv")

x = dataset.values.reshape(-1,1).astype(np.float64)
print(x)
y = dataset.values.reshape(-1,1).astype(np.float64)
print(y)

inputs = Input(shape=(1,))
preds = Dense(1,activation='linear')(inputs)

# min-max -1,1
sc = MinMaxScaler(feature_range=(-1,1))
x_ = sc.fit_transform(x)
y_ = sc.fit_transform(y)

model = Model(inputs=inputs,outputs=preds)
sgd=keras.optimizers.SGD(lr = 0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd ,loss='mse')
model.fit(x_,y_, batch_size=1, verbose=1, epochs=100, shuffle=True)
plt.scatter(x_,y_,color='black')
plt.plot(x_,model.predict(x_), color='blue', linewidth=3)
plt.show()