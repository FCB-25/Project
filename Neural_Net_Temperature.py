from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from numpy.random import seed
import tensorflow as tf
import util as u
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_model(new_model):
    if(new_model):
        # create model
        model = Sequential()
        model.add(Dense(512, input_dim=29, kernel_initializer='normal', activation='relu'))
        model.add(Dense(256,kernel_initializer='normal', activation='sigmoid'))
        model.add(Dense(29,kernel_initializer='normal', activation='tanh'))
        model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))

        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    else:
        # load json and create model
        json_file = open('model2.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model2.h5")
        print("Loaded model from disk")
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    return model

""" load and prepare Data """

colnames = ['RowID', 'STATIONS_ID', 'MESS_DATUM', 'V_N_x', 'V_S1_CS', 'V_S1_HHS', 'V_S1_NS', 'V_S2_CS', 'V_S2_HHS',
            'V_S2_NS', 'V_N_y', 'P', 'P0', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D']

result = pd.read_csv('results_clean_medium.csv', names=colnames, sep=',', header=1)

result.drop('RowID', 1, inplace=True)
result.drop('STATIONS_ID', 1, inplace=True)
result.drop('MESS_DATUM', 1, inplace=True)
result.drop('V_N_y', 1, inplace=True)
result.drop('P', 1, inplace=True)
result.drop('P0', 1, inplace=True)



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
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=0)

model = load_model(False)


model.fit(X_train, y_train, epochs=1, batch_size=50,  verbose=1, validation_split=0.2)


x = X_test[0:10]
y = y_test[0:10]
# Make predictions using the testing set
y_pred = model.predict(x)


print(y)
print(y_pred)

# serialize model to JSON
model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
print("Saved model to disk")