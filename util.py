from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# copy pasted submission 3 ex1
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single column with index `key` from some matrix X"""

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self  # do nothing during fitting procedure

    def transform(self, data_matrix):
        return data_matrix[:, [self.key]]  # return a matrix with single column

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Assumes that input X to fit and transform is a single
    column matrix of categorical values."""
    def fit(self, X, y=None):
        # determine unique labels
        self.elements = np.unique(X[:, 0])
        self.elements.sort()
        return self

    def transform(self, X, y=None):
        return np.column_stack([X[:,0] == e for e in self.elements])*1.0