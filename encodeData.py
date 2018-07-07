import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.ensemble import GradientBoostingClassifier


# copy pasted submission 3 ex1
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single column with index `key` from some matrix X"""

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self  # do nothing during fitting procedure

    def transform(self, data_matrix):
        return data_matrix[:, [self.key]]  # return a matrix with single column


if __name__ == '__main__':

    colnames = ['RowID', 'STATIONS_ID', 'MESS_DATUM', 'V_N_x', 'V_S1_CS', 'V_S1_HHS', 'V_S1_NS', 'V_S2_CS', 'V_S2_HHS',
                'V_S2_NS', 'V_N_y', 'P', 'P0', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D']

    result = pd.read_csv('results2.csv', names=colnames, sep=',', header=1)

    result.drop('RowID', 1, inplace=True)

    V_S1_CS = result.get('V_S1_CS').astype(str)  # [0,9]
    V_S2_CS = result.get('V_S2_CS').astype(str)  # [0,9]
    RS_IND = result.get('RS_IND').astype(str)  # [0,1]

    XY = result.values

    X, y = XY, XY[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,test_size=0.25, random_state=0)





    features = make_union(
       # make_pipeline(ColumnSelector(3), OneHotEncoder()),
       # make_pipeline(ColumnSelector(6), OneHotEncoder()),
        make_pipeline(ColumnSelector(13), OneHotEncoder()),
    )


    svc_pipeline = GridSearchCV(
        estimator=make_pipeline(features, StandardScaler(with_mean=False), SVC()),
        param_grid=[{
            'svc__C': 10 ** np.arange(-6.0, 6.0, 1.0),
            'svc__gamma': 10 ** np.arange(-6.0, 1.0, 1.0),
        }],
        verbose=2,
        n_jobs=-1
    )

    boosting_pipeline = GridSearchCV(
        estimator=make_pipeline(features, StandardScaler(with_mean=False), GradientBoostingClassifier()),
        param_grid=[{
            'gradientboostingclassifier__n_estimators': [2 ** i for i in range(11)],
            'gradientboostingclassifier__learning_rate': 10 ** np.arange(-5.0, 1.0, 1.0),
        }],
        verbose=2,
        n_jobs=-1
    )

    dummy_pipeline = make_pipeline(features, DummyClassifier('most_frequent'))

    all_models = {
        #'GradientBoostingClassifier': boosting_pipeline,
        'SVC': svc_pipeline,
        'DummyClassifier': dummy_pipeline
    }

    results = {}

    for name, pipe in all_models.items():
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        results[name] = score

    best_score = 0.0
    print("scores:")
    for name, score in results.items():
        print(name + ": " + str(score))
        # save best model
        if best_score < score:
            best_score = score
            best_model = all_models[name]

    import pickle as pc

    pc.dump(best_model, open('model.bin', 'wb'))
