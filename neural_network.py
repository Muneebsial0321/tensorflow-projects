

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


texts = [
    "I love machine learning",
    "Deep learning is great",
    "Natural language processing is fascinating",
    "I dislike spam emails",
    "Machine learning is awesome",
    "I hate spam", "A bad tempered person", "chess is waist of time",
    "Artificial intelligence is the future", "Pactice makes man perfect"
]
labels = [1, 1, 1, 0, 1, 0, 0,0, 1, 1]  



# Parameters
max_words = 10000  # Maximum number of words in the vocabulary
max_len = 50  # Maximum length of input sequences





tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)



X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)




  

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(Flatten())
model.add(Dense(64, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))  







# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  
              metrics=['accuracy'])






# Train the model
history = model.fit(X, y, epochs=5, batch_size=2, validation_split=0.2) 
history_dict = history.history






# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")






acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()   




plt.plot(epochs, acc, 'bo', label='Training acc')
#plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()