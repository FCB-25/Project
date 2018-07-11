import Neural_Net
import pandas as pd
import Model_Tester as t
from sklearn.model_selection import train_test_split

colnames = ['RowID', 'V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS', 'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO',
            'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3', 'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2',
            '3', '4', '5', '6', '7', '8', '1H_RS_IND']

dense = 30
modelname = "model_nextH"
result = pd.read_csv("results_medium_next_hour.csv", names=colnames, sep=',', header=1)
result.drop('RowID', 1, inplace=True)

XY = result.values
sel = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
idx_y = -1

XY = result.values

X_train, X_test = train_test_split(XY, train_size=0.7, test_size=0.3, random_state=0)

X, y = XY[:, sel], XY[:, idx_y]
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=0)


model = Neural_Net.neural_net.fit_lstm(X_train, 2, 1, 29)

print(model)
testData = pd.read_csv('testData.csv', names=colnames, sep=',', header=1)
testData.drop('RowID', 1, inplace=True)

XY = testData.values

sel = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
X, y = XY[:, sel], XY[:, -1]

t.Tester.test_model(model, X, y)
