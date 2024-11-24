import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# Load IMDB dataset
max_features = 10000  
maxlen = 100          
batch_size = 30


(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)

# Pad sequences (ensure that all sequences are of the same length)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)




# model


model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=10, input_length=maxlen))  # Corrected output_dim
model.add(LSTM(64, return_sequences=True))  
model.add(LSTM(64))  
# model.add(LSTM(46))  
model.add(Dense(46, activation='softmax'))  


model.compile(loss='sparse_categorical_crossentropy',  
              optimizer='adam',  
              metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=150, epochs=40)



# Evaluate the model
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f'Test score: {score}')
print(f'Test accuracy: {acc}')