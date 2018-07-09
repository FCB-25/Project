from Neural_Net import neural_net
import Model_Tester as t
import pandas as pd
import pickle as pc

colnames = ['RowID', 'V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS',
            'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3',
            'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2', '3', '4', '5', '6', '7', '8']

colnames_2 = ['RowID', 'V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS',
              'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3',
              'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2', '3', '4', '5', '6', '7', '8', '1H_RS_IND']

dense = 29
dense_2 = 30

modelname = "model.h5"
modelname_2 = "model_nextH.h5"

filename = "results_V2_extended.csv"
filename_2 = "results_medium_next_hour.csv"

epochs = 200

result = pd.read_csv(filename, names=colnames, sep=',', header=1)
result.drop('RowID', 1, inplace=True)

XY = result.values

sel_1 = [x for x in range(XY.shape[1]) if x != 6]
idx_y_1 = 6

sel_2 = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
idx_y_2 = -1

if '1H_RS_IND' in result.head(0):
    RS_IND = result.get('1H_RS_IND').astype(str)
    result.drop('1H_RS_IND', 1, inplace=True)
    RS_IND = pd.DataFrame(RS_IND, columns=['1H_RS_IND'])
    result = result.join(RS_IND)

# run neural network
model = neural_net.nn(modelname, dense, result, epochs, sel=sel_1, idx_y=idx_y_1)

if '1H_RS_IND' in result.head(0):
    # test neural network on special data
    testData = pd.read_csv('testData.csv', names=colnames_2, sep=',', header=1)
    testData.drop('RowID', 1, inplace=True)

    XY = testData.values

    sel = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
    X, y = XY[:, sel], XY[:, -1]

    t.Tester.test_model(model, X, y)

"""
# test Tree
dt = pc.load(open('modeldt_1H.bin','rb'))

colnames_2 = ['V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS',
              'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3',
              'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2', '3', '4', '5', '6', '7', '8', '1H_RS_IND']

t.Tester.test_dt(dt,colnames_2)
"""
