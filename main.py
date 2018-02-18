import sys
sys.path.append('experiments/' + sys.argv[1])

from feature_extractor import FeatureExtractor
from classifier import Classifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

le = preprocessing.LabelEncoder()


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


# load data
X_train, y_train = load_data('data/train.csv')
X_valid, y_valid = load_data('data/valid.csv', test=True)

print "Data statistics"

print X_train.shape, X_valid.shape

print len(y_valid[y_valid == 0]) / float(len(y_valid))
print len(y_valid[y_valid == 1]) / float(len(y_valid))
print len(y_valid[y_valid == 2]) / float(len(y_valid))

print len(y_train[y_train == 0]) / float(len(y_train))
print len(y_train[y_train == 1]) / float(len(y_train))
print len(y_train[y_train == 2]) / float(len(y_train))


# preprocess data
feature_extractor = FeatureExtractor()
X_train = feature_extractor.fit_transform(X_train)
X_valid = feature_extractor.transform(X_valid)

# model
clf = Classifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)

# predict
print accuracy_score(y_valid, y_pred)
