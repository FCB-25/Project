from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union


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

""" load and prepare Data """

colnames = ['RowID', 'STATIONS_ID', 'MESS_DATUM', 'V_N_x', 'V_S1_CS', 'V_S1_HHS', 'V_S1_NS', 'V_S2_CS', 'V_S2_HHS',
            'V_S2_NS', 'V_N_y', 'P', 'P0', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D']

result = pd.read_csv('results_medium.csv', names=colnames, sep=',', header=1)

result.drop('RowID', 1, inplace=True)

V_S1_CS = result.get('V_S1_CS').astype(str)  # [0,9]
V_S2_CS = result.get('V_S2_CS').astype(str)  # [0,9]
RS_IND = result.get('RS_IND').astype(str)  # [0,1]

XY = result.values

sel = ColumnSelector(3).transform(XY)
sel2 = ColumnSelector(6).transform(XY)

coder = OneHotEncoder()

coder = coder.fit(sel)
sel = coder.transform(sel)

coder = coder.fit(sel2)
sel2 = coder.transform(sel2)
colnames = ['sel_0','sel_1','sel_2','sel_3','sel_4','sel_5','sel_6','sel_7','sel_8']

result = result.join(pd.DataFrame(sel,columns=colnames ))
result = result.join(pd.DataFrame(sel2))
result.drop('V_S1_CS',1, inplace=True)
result.drop('V_S2_CS',1, inplace=True)

XY = result.values

sel = [x for x in range(XY.shape[1]) if x != 11]
X, y = XY[:, sel], XY[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

# create model
model = Sequential()
model.add(Dense(512, input_dim=34, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(34, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=50, batch_size=10)

# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

