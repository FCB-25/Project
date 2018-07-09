import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

""" load and prepare Data """
colnames = ['RowID', 'STATIONS_ID', 'MESS_DATUM', 'V_N_x', 'V_S1_CS', 'V_S1_HHS', 'V_S1_NS', 'V_S2_CS', 'V_S2_HHS',
            'V_S2_NS', 'V_N_y', 'P', 'P0', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D']

result = pd.read_csv('results_medium.csv', names=colnames, sep=',', header=1)
# result = pd.read_csv('results.csv', names=colnames, sep=',', header=1)

result.drop('RowID', 1, inplace=True)
result.drop('STATIONS_ID', 1, inplace=True)
result.drop('MESS_DATUM', 1, inplace=True)

V_S1_CS = result.get('V_S1_CS').astype(str)  # [0,9]
V_S2_CS = result.get('V_S2_CS').astype(str)  # [0,9]
RS_IND = result.get('RS_IND').astype(str)  # [0,1]

XY = result.values
sel = [x for x in range(XY.shape[1]) if x != 11]
X, y = XY[:, sel], XY[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

clf = svm.SVC(kernel='linear', C=1.0)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

#pc.dump(model, open('modelsvc.bin', 'wb'))
