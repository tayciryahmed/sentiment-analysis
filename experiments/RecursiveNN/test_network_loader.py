import sys
import torch
import itertools
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
le = preprocessing.LabelEncoder()

from utils import plot_confusion_matrix

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

def accuracy_per_class(y, y_pred, classes):
    n_classes = len(classes)
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    # (real, predicted)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i in range(y.shape[0]):
        label = y[i]
        confusion_matrix[label, y_pred[i]] += 1
        if y[i] == y_pred[i]:
            class_correct[label] += 1
        class_total[label] += 1

    print("{:<10} {:^10}".format("Class", "Accuracy (%)"))
    for i in range(n_classes):
        print('{:<10} {:^10.2f}'.format(
            classes[i], 100 * class_correct[i] / class_total[i]))
    return confusion_matrix


X_train, y_train = load_data('../../data/train.csv')
X_valid, y_valid = load_data('../../data/valid.csv', test=True)

train_tokens = pd.Series(X_train).apply(clean_text_simple)
train_data = tr.loadTrees(train_tokens,y_train)
train_sents = [tree.get_words() for tree in train_data]
          
vocab = Vocab()
vocab.construct(list(itertools.chain.from_iterable(train_sents)))
        
test_net = RNN_Model(vocab,embed_size = 300)

for model in range(29,30):
    model_path = "poids_entrainement/recursiveNN_model_epoch{}.pth".format(model)
    test_net.load_state_dict(torch.load(model_path))

    y_pred_train = predict(X_train,test_net)
    print("Train: iter", model,  accuracy_score(y_train, y_pred_train))

    classes = ('negative', 'neutral', 'positive')
    confusion_matrix_train = accuracy_per_class(y_train, y_pred_train, classes)
    # Plot normalized confusion matrix
    plot_confusion_matrix(confusion_matrix_train, classes, normalize=True,
                      title='Normalized confusion matrix, train')

    y_pred = predict(X_valid,test_net)
    print("Val: iter", model, accuracy_score(y_valid, y_pred))
    confusion_matrix_pred = accuracy_per_class(y_valid, y_pred, classes)
    # Plot normalized confusion matrix
    plot_confusion_matrix(confusion_matrix_pred, classes, normalize=True,
                      title='Normalized confusion matrix, val')