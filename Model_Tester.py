import pandas as pd
import numpy as np


class Tester:

    def test_model(model, X, y):
        scores = model.evaluate(X, y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        y_pred = model.predict(X)
        x = np.zeros(y_pred.shape)

        for i in range(0, y_pred.size):
            x[i] = y[i] - y_pred[i]
        print(x)

    def test_dt(model, colnames):
        testData = pd.read_csv('testData.csv', names=colnames, sep=',', header=1)

        XY = testData.values

        sel = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
        X, y = XY[:, sel], XY[:, -1]

        y_pred = model.predict(X)
        x = np.zeros(y_pred.shape)
        for i in range(0, y_pred.size):
            x[i] = y[i] - y_pred[i]
        print(x)

        print("accuracy: " + str(model.score(X, y) * 100) + "%")
