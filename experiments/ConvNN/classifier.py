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
from keras.layers import Dense,  Dropout
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
    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in punct)
    text = re.sub(' +', ' ', text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space
    tokens = text.split()  # tokenize (split based on whitespace)
    tokens = [w for w in tokens if w.isalpha()] # keep words with alphabetic characters
    tokens = [w for w in tokens if len(w) > 1]  # remove words with lenght=1

    if remove_stopwords:
        # remove stopwords from 'tokens'
        tokens = [x for x in tokens if x not in my_stopwords]

    if stemming:
        # apply stemmer
        stemmer = SnowballStemmer('english')
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens

# Function to create model, required for KerasClassifier
def create_model1(max_length=500, vocab_size=500):
    # create model
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=max_length))
    model.add(Conv1D(filters=256, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[metrics.categorical_accuracy])
    return model

# Function to create model, required for KerasClassifier


class Classifier():
    def __init__(self):
        self.min_occur = 1
        self.max_length = -1
        self.vocab_size = -1
        self.tokenizer = Tokenizer()  # create the tokenizer

    def fit(self, X, y):

        ytrain = np.zeros((y.shape[0], 3))
        for i in range(y.shape[0]):
            ytrain[i, y[i]] = 1

        statements = pd.Series(X).apply(clean_text_simple)

        vocab = Counter()
        for statement in statements:
            vocab.update(statement)
        tokens = [k for k, c in vocab.items() if c >= self.min_occur]
        statements = statements.apply(lambda x: [w for w in x if w in tokens])

        statements = statements.apply(lambda x: ' '.join(x))
        train_statements = list(statements.values)

        # fit the tokenizer on the statements
        self.tokenizer.fit_on_texts(train_statements)
        self.max_length = max([len(s.split()) for s in train_statements])
        self.vocab_size = len(self.tokenizer.word_index) + 1


        encoded_statements = self.tokenizer.texts_to_sequences(
            train_statements)
        Xtrain = pad_sequences(
            encoded_statements, maxlen=self.max_length, padding='post')

        self.clf2 = KerasClassifier(
            build_fn=create_model1, max_length=self.max_length, vocab_size=self.vocab_size, epochs=12)
        self.clf = KerasClassifier(build_fn=create_model1, max_length=self.max_length, vocab_size=self.vocab_size, epochs=12)

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
        Xtest = pad_sequences(encoded_statements,
                              maxlen=self.max_length, padding='post')
        y1 = self.clf.predict_proba(Xtest)
        y2 = self.clf2.predict_proba(Xtest)

        y = np.where(np.repeat((np.max(y1, axis=1) > np.max(
            y2, axis=1)).reshape(y1.shape[0], 1), 3, axis=1), y1, y2)

        return y
