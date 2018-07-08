import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle as pc

""" load and prepare Data """
colnames = ['RowID', 'STATIONS_ID', 'MESS_DATUM', 'V_N_x', 'V_S1_CS', 'V_S1_HHS', 'V_S1_NS', 'V_S2_CS', 'V_S2_HHS',
            'V_S2_NS', 'V_N_y', 'P', 'P0', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D']

#result = pd.read_csv('results_medium.csv', names=colnames, sep=',', header=1)
result = pd.read_csv('results.csv', names=colnames, sep=',', header=1)

result.drop('RowID', 1, inplace=True)

V_S1_CS = result.get('V_S1_CS').astype(str)  # [0,9]
V_S2_CS = result.get('V_S2_CS').astype(str)  # [0,9]
RS_IND = result.get('RS_IND').astype(str)  # [0,1]

XY = result.values
sel = [x for x in range(XY.shape[1]) if x != 13]
X, y = XY[:, sel], XY[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

model = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid={
        'max_depth': [2 ** i for i in range(1, 8)],
        'min_samples_split': [2.0 ** (-i) for i in range(2, 11)],
    }
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

pc.dump(model, open('modeldt.bin', 'wb'))
