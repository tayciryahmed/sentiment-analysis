# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import string
import re
import zipfile
import io

from collections import Counter

from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense ,  Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import metrics
from gensim.models import Word2Vec
import requests



stpwrds = set([stopword for stopword in stopwords.words('english')])
punct = set(string.punctuation.replace('-', ''))
punct.update(["``", "`", "..."])



# Tokenize
def clean_text_simple(text, my_stopwords=stpwrds, punct=punct, remove_stopwords=True, stemming=False):
	text = text.lower()
	text = ''.join(l for l in text if l not in punct) # remove punctuation (preserving intra-word dashes)
	text = re.sub(' +',' ',text) # strip extra white space
	text = text.strip() # strip leading and trailing white space
	tokens = text.split() # tokenize (split based on whitespace)
	tokens = [w for w in tokens if w.isalpha()]
	tokens = [w for w in tokens if len(w) > 1]

	if remove_stopwords:
		# remove stopwords from 'tokens'
		tokens = [x for x in tokens if x not in my_stopwords]

	if stemming:
		# apply stemmer
		stemmer = SnowballStemmer('english')
		tokens = [stemmer.stem(t) for t in tokens]

	return tokens

# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	#lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
	return embedding

def load_embedding(filename):
	# load embedding into memory, skip first line
	#url_opened = urllib.request.urlopen(url)
	#zipped = zipfile.ZipFile(io.BytesIO(url_opened.read()))

	lines = open(filename).readlines()[1:]
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0].decode('utf-8') ] = np.asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, 300))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix

# Function to create model, required for KerasClassifier
def create_model1(max_length=500, vocab_size=500):
	# create model
	model = Sequential()
	model.add(Embedding(vocab_size, 100, input_length=max_length))
	model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(20, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
	return model

# Function to create model, required for KerasClassifier
def create_model2(embedding_vectors=[], max_length=0, vocab_size=0):
	# create the embedding layer
	embedding_layer = Embedding(vocab_size, 300, weights=embedding_vectors, input_length=max_length, trainable=False)
	# define model
	model = Sequential()
	model.add(embedding_layer)
	model.add(Conv1D(filters=256, kernel_size=8, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(20, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
	return model



class Classifier():
	def __init__(self):
		self.min_occur = 1
		self.max_length = -1
		self.vocab_size = -1
		self.tokenizer = Tokenizer() # create the tokenizer
		#self.raw_embedding = load_embedding(filename='glove/glove.6B.200d.txt')
		self.raw_embedding = load_embedding(filename='experiments/keras_glove/glove.6B.300d.txt')
		self.metaclf = RandomForestClassifier()

	def fit(self, X, y):

		ytrain = np.zeros((y.shape[0], 3))
		for i in range(y.shape[0]):
			ytrain[i, y[i]] = 1

		statements = pd.Series(X).apply(clean_text_simple)

		vocab = Counter()
		for statement in statements:
			vocab.update(statement)
		tokens = [k for k,c in vocab.items() if c >= self.min_occur]
		statements = statements.apply(lambda x: [w for w in x if w in tokens])

		statements = statements.apply(lambda x: ' '.join(x))
		train_statements = list(statements.values)

		self.tokenizer.fit_on_texts(train_statements) # fit the tokenizer on the statements
		self.max_length = max([len(s.split()) for s in train_statements])
		self.vocab_size = len(self.tokenizer.word_index) + 1

		# get vectors in the right order
		embedding_vectors = get_weight_matrix(self.raw_embedding, self.tokenizer.word_index)

		encoded_statements = self.tokenizer.texts_to_sequences(train_statements)
		Xtrain = pad_sequences(encoded_statements, maxlen=self.max_length, padding='post')

		self.clf2 = KerasClassifier(build_fn=create_model1, max_length=self.max_length, vocab_size=self.vocab_size, epochs=12)
		self.clf = KerasClassifier(build_fn=create_model2, embedding_vectors=[embedding_vectors], max_length=self.max_length, vocab_size=self.vocab_size, epochs=12)


		self.clf.fit(Xtrain, ytrain, epochs=12)
		self.clf2.fit(Xtrain, ytrain, epochs=12)


	def predict(self, X):
		y_proba = self.predict_proba(X)
		y = np.argmax(y_proba, axis=1)
		return y

	def predict_proba(self, X):
		statements = pd.Series(X).apply(clean_text_simple)
		statements = statements.apply(lambda x: ' '.join(x))
		statements = list(statements.values)
		encoded_statements = self.tokenizer.texts_to_sequences(statements)
		Xtest = pad_sequences(encoded_statements, maxlen=self.max_length, padding='post')
		y1 = self.clf.predict_proba(Xtest)
		y2 = self.clf2.predict_proba(Xtest)

		y = np.where(np.repeat((np.max(y1, axis=1)>np.max(y2, axis=1)).reshape(y1.shape[0], 1), 3, axis=1), y1, y2)

		return y
