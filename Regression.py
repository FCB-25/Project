import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import seed
import util as u
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm,linear_model
import pickle as pc


seed(1)
""" load and prepare Data """

colnames = ['RowID', 'STATIONS_ID', 'MESS_DATUM', 'V_N_x', 'V_S1_CS', 'V_S1_HHS', 'V_S1_NS', 'V_S2_CS', 'V_S2_HHS',
            'V_S2_NS', 'V_N_y', 'P', 'P0', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D']

result = pd.read_csv('results_clean_medium.csv', names=colnames, sep=',', header=1)

result.drop('RowID', 1, inplace=True)
result.drop('STATIONS_ID', 1, inplace=True)
result.drop('MESS_DATUM', 1, inplace=True)
result.drop('V_S1_HHS', 1, inplace=True)
result.drop('V_S2_HHS', 1, inplace=True)
result.drop('V_N_x', 1, inplace=True)



V_S1_CS = result.get('V_S1_CS').astype(str)  # [0,9]
V_S2_CS = result.get('V_S2_CS').astype(str)  # [0,9]
RS_IND = result.get('RS_IND').astype(str)  # [0,1]

XY = result.values

sel = u.ColumnSelector(1).transform(XY)
sel2 = u.ColumnSelector(4).transform(XY)

coder = u.OneHotEncoder()

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

sel = [x for x in range(XY.shape[1]) if x != 7]
X, y = XY[:, sel], XY[:, 7]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=0)

model = linear_model.Ridge()

model.fit(X_train,y_train)

print(y)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

print(model.predict(x))