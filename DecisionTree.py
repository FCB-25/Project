import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle as pc
import Project.Model_Tester as t
import Project.util as u

colnames = ['V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS',
            'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3',
            'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2', '3', '4', '5', '6', '7', '8','1H_RS_IND']

result = pd.read_csv('results_medium_next_hour.csv', names=colnames, sep=',', header=1)

XY = result.values

sel = [x for x in range(XY.shape[1]) if x != 6]
X, y = XY[:, sel], XY[:, 6]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=0)

model = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid={
        'max_depth': [2 ** i for i in range(1, 8)],
        'min_samples_split': [2.0 ** (-i) for i in range(2, 11)],
    }
)

print("start fitting model...")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

pc.dump(model, open('modeldt_1H.bin', 'wb'))