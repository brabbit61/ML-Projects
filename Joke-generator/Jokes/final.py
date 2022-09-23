from __future__ import print_function
import numpy as np
import csv
from keras.models import Sequential, model_from_yaml,save_model,load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
	
	
'''
def sample(preds, temperature=1.0):
	print(preds.shape)
	preds = np.asarray(preds).astype('float64')
	print(preds.shape)
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds/temperature)
	preds = exp_preds / np.sum(exp_preds)
	return np.argmax(preds)

def generate_text_posttraining(model,vocablen,ix_to_char,char_to_ix):
	global temperature
	word=str(raw_input("Enter a word to start the joke: "))
	prefixes=['A ','The ','All the ', 'Did you know a ']
	index=np.random.randint(0,len(prefixes))
	word= prefixes[index]+word
	length=np.random.randint(50,150)
	print(length)
	y_char=[word]
	ix=[char_to_ix[y_char[0][-1]]]
	X = np.zeros((1, length, vocablen))
	for i in range(length):
		X[0, i, :][ix[-1]] = 1						
		preds = model.predict(X[:, :i+1, :])[0]
        ix = sample(preds, temperature)
        next_char = ix_to_char[ix[-1]]
        print(model.predict(X[:, :i+1, :])[0].shape)
        y_char.append(next_char)
	y_char.append(" end")
	return ('').join(y_char)	
'''

def generate_text(model, vocablen, ix_to_char):
	#length=np.random.randint(70,200)
	length=200
	ix = [np.random.randint(vocablen)]
	y_char = [ix_to_char[ix[-1]]]
	X = np.zeros((1, length, vocablen))
	for i in range(length):
		X[0, i, :][ix[-1]] = 1
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
	print(('').join(y_char))
	
def preprocessing(data):
	X = np.zeros((len(data)/seq_length, seq_length, vocablen))
	y = np.zeros((len(data)/seq_length, seq_length, vocablen))
	for i in range(0, len(data)/seq_length):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		input_sequence = np.zeros((seq_length, vocablen))
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
			X[i] = input_sequence
		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, vocablen))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
		return X,y
		

temperature=1.5
batch_size=200
layer_num=3
seq_length=50
hidden_dim=450
epochs=5

data=open("funnytweeter.csv", 'r').read()
data+=open("reddit-cleanjokes.csv", 'r').read()

chars=set(data)
vocablen=len(chars)
ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}


model = Sequential()
model.add(LSTM(hidden_dim, input_shape=(None, vocablen), return_sequences=True))
for i in range(layer_num - 1):
  model.add(LSTM(hidden_dim, return_sequences=True,dropout=0.35))
model.add(TimeDistributed(Dense(vocablen)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer='rmsprop')

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
model=load_model('final3.hdf5')
'''
count=0

while epochs:
	print( "\n\nEpoch: ",count)
	count+=1
	epochs-=1
	
	X,y=preprocessing(data)
	model.fit(X, y, batch_size=batch_size, verbose=1, epochs=1)
	
	generate_text(model, vocablen, ix_to_char)
	
model.save('finale'+str(epochs)+'.hdf5')
'''
