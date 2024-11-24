import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



data = keras.datasets.mnist
data = data.load_data()



(xtrain,ytrain),(xtest,ytest)= data
ytrain[0]
xtrain= xtrain/255.00
xtest= xtest/255.00
plt.imshow(xtrain[7],cmap=plt.cm.binary)





modal = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])




modal.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


his =modal.fit(xtrain,ytrain,epochs=10, verbose=2)



loss,acc= modal.evaluate(xtest,ytest)
acc
his_ = his.history

print("accuracy is ",acc)


pred = modal.predict(xtest)





acc = his_['accuracy']
epochs = range(1, len(acc) + 1)
loss = his_['loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
