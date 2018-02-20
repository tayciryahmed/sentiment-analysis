import sys
import torch
import itertools
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
le = preprocessing.LabelEncoder()

sys.path.append('experiments/RecursiveNN')

from classifier import RNN_Model
from utils import Vocab, clean_text_simple
import tree_class as tr

def load_data(filename, test=False):

    df = pd.read_csv(filename, sep='\t', header=None, usecols=[2, 3],
                     names=['class', 'text'])

    df.loc[df['class'] == 'objective-OR-neutral','class'] = 'neutral'
    df.loc[df['class'] == 'objective','class'] = 'neutral'

    X = df['text']
    y = df['class']

    if not test:
        y = le.fit_transform(y)
    else:
        y = le.transform(y)

    return X, y

def predict(X,model):
    y_proba = predict_proba(X,model)
    y = np.argmax(y_proba, axis=1)
    return y

def predict_proba(X,model):
    y = np.empty((0,3))
    tokens = pd.Series(X).apply(clean_text_simple)
    data = [tr.Tree(tokens[i],None) for i in range(tokens.shape[0])]
    #model.load_state_dict(torch.load("model.pth"))
    for step, tree in enumerate(data): 
        all_nodes_prediction = model(tree)
        root_prediction = all_nodes_prediction[-1]
        y = np.append(y, root_prediction.data.numpy(), axis=0)
    return y


X_train, y_train = load_data('data/train.csv')
X_valid, y_valid = load_data('data/valid.csv', test=True)

train_tokens = pd.Series(X_train).apply(clean_text_simple)
train_data = tr.loadTrees(train_tokens,y_train)
train_sents = [tree.get_words() for tree in train_data]
          
vocab = Vocab()
vocab.construct(list(itertools.chain.from_iterable(train_sents)))
        
test_net = RNN_Model(vocab,embed_size = 300)
model_path = "experiments/Recursive NN/trainings/model_epoch17.pth"
test_net.load_state_dict(torch.load(model_path))

y_pred_train = predict(X_train,test_net)
print("Train:", accuracy_score(y_train, y_pred_train))

y_pred = predict(X_valid,test_net)
print("Val:", accuracy_score(y_valid, y_pred))