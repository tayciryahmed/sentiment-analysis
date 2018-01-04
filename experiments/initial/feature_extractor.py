# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from sklearn.feature_extraction.text import TfidfVectorizer


def document_preprocessor(doc):
    return doc


def token_processor(tokens):
    for t in tokens:
        yield t

class FeatureExtractor(TfidfVectorizer):

    def __init__(self):
        super(FeatureExtractor, self).__init__(
                analyzer='word', preprocessor=document_preprocessor)

    def fit(self, X_df, y=None):
        super(FeatureExtractor, self).fit(X_df)
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        X = super(FeatureExtractor, self).transform(X_df)
        return X

    def build_tokenizer(self):
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
