# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import string


class FeatureExtractor():
	def __init__(self):
		pass

	def fit(self, X_df, y=None):
		return self

	def fit_transform(self, X_df, y=None):
		self.fit(X_df)
		return self.transform(X_df)

	def transform(self, X_df):
		return X_df
