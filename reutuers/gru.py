import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense ,SimpleRNN,Convolution2D,Dropout
from tensorflow.keras.utils import to_categorical

# Load the Reuters dataset

max_len = 150  
max_words = 5000  

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words)

# Pad sequences to ensure consistent input length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
x_test =x_test[:1000]

# Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, num_classes=46)
y_test = to_categorical(y_test, num_classes=46)
y_test =y_test[:1000]
# x_train= x_train[:3000]
# y_train= y_train[:3000]
# x_test=x_train[3000:4000]
# y_test=y_train[3000:4000]




model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=46, input_length=max_len))
model.add(GRU(256, return_sequences=False))  # Increased GRU units
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))  # Output layer with softmax activation



# model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=100)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')





