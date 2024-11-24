
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



data = pd.read_csv('./IMDB Dataset.csv')
df= pd.DataFrame(data)
df= df.replace(['positive','negative'],[1,0])
x=df['review'][0:1000]
y=df['sentiment'][0:1000]
print(y)




tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x)



padded_sequences = pad_sequences(sequences, padding='post', maxlen=50)



X = pad_sequences(sequences, maxlen=500)
y = np.array(y)  
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=.3)




# Build the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=500))
model.add(Flatten())    
model.add(Dense(64, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy', 
              metrics=['accuracy'])



padded_sequences = np.array(padded_sequences)
history =  model.fit(padded_sequences, np.array(y), epochs=3, verbose=2) 
loss, accuracy = model.evaluate(padded_sequences,np.array(y))
print(f'Accuracy: is {accuracy:.4f}')
history_dict = history.history



print(f'Accuracy: is {accuracy:.4f}')
acc = history_dict['accuracy']



epochs = range(1, len(acc) + 1)
loss = history_dict['loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, acc, label='Validation acc',color='red')


