import pandas as pd

class Tester:

    def test_model(model,X,y):
        scores = model.evaluate(X, y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    def test_dt(model,colnames):
        testData = pd.read_csv('testData.csv', names=colnames, sep=',', header=1)

        XY = testData.values

        sel = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
        X, y = XY[:, sel], XY[:, -1]

        print(model.score(X, y))
