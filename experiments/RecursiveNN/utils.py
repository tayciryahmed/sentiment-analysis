import numpy as np
import string
import re

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import matplotlib.pyplot as plt
import itertools

class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print('{} total words with {} uniques'.format(self.total_words, len(self.word_freq)))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)
    
stpwrds = set([stopword for stopword in stopwords.words('english')])
punct = set(string.punctuation.replace('-', ''))
punct.update(["``", "`", "..."])  
def clean_text_simple(text, my_stopwords=stpwrds, punct=punct, remove_stopwords=True, stemming=False):
    # text is a sentence 
    
    # put text to lower case 
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in punct)
    text = re.sub(' +', ' ', text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space
    tokens = text.split()  # tokenize (split based on whitespace)
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

def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename, 'r')
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

def arrange_embeddings(embeddings, vocab, embed_size):
    # total vocabulary size plus 0 for unknown words
    vocab_size = vocab.__len__()
    # define weight matrix dimensions with all 0
    arranged_embeddings = np.zeros((vocab_size, embed_size))
    for index in range(vocab_size):
        word = vocab.decode(index)
        vector = embeddings.get(word)
        if vector is not None:
            arranged_embeddings[index] = vector
    return arranged_embeddings

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    :param cm: (numpy matrix) confusion matrix
    :param classes: [str]
    :param normalize: (bool)
    :param title: (str)
    :param cmap: (matplotlib color map)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')