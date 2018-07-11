from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle as pc



class DecisionTree:
    def run_decisionTree(data):
        XY = data.values

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
        return model
