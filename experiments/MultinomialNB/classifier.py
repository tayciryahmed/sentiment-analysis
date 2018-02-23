# -*- coding: utf-8 -*-

from sklearn.naive_bayes import MultinomialNB


class Classifier():
    def __init__(self):
        self.clf = MultinomialNB()

    def fit(self, X, y):
        self.clf.fit(X.todense(), y)

    def predict(self, X):
        return self.clf.predict(X.todense())

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
