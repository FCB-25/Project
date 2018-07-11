from Neural_Net import neural_net
import Model_Tester as t
import DecisionTree as tree
import pandas as pd
import sys
import pickle as pc

print("every boolean inputs must be 1 (True) or 0 (False)")

predict = input("Do you want to make predictions or train a model? ")

while predict != 'train' and predict != 'predict':
    predict = input("Your choice was not clear... predict or train? ")

if predict == 'train':
    dataset = str(input("Which dataset should be used? "))
    if dataset != 'results_medium_next_hour.csv' and dataset != 'results_V2_extended.csv':
        dataset = str(input("This is no valid dataset... do you mean results_V2_extended.csv? "))
        if dataset == '1' or dataset == 'yes':
            dataset = "results_V2_extended.csv"

    if dataset != 'results_medium_next_hour.csv' and dataset != 'results_V2_extended.csv':
        dataset = str(input("Do you mean results_medium_next_hour.csv? "))
        if dataset == '1' or dataset == 'yes':
            dataset = "results_medium_next_hour.csv"

    while dataset != 'results_V2_extended.csv' and dataset != 'results_medium_next_hour.csv':
        dataset = str(input(
            "PLease type the right name again.. (make sure the dataset is in the directory like this class and you typed the right name) "))

    run_neural_net = input("Do you want to run a Neural Net? (otherwise a Decision Tree will be executed) ")
    while run_neural_net != '1' and run_neural_net != '0':
        run_neural_net = input("This is no valid input... please type 1 for executing a neural net or 0 otherwise ")

    new_model = input("If you want to create a new model, type 'yes' ")
    if new_model == 'yes':
        new_model = True
    else:
        new_model = False

    if run_neural_net == '1':
        print("load data...")
        # use dataset for prediction, if it's raining now
        if dataset == 'results_V2_extended.csv' or dataset == '1':
            colnames = ['RowID', 'V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS',
                        'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3',
                        'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2', '3', '4', '5', '6', '7', '8']
            dense = 29
            modelname = "model"

            result = pd.read_csv(dataset, names=colnames, sep=',', header=1)
            result.drop('RowID', 1, inplace=True)

            XY = result.values

            sel = [x for x in range(XY.shape[1]) if x != 6]
            idx_y = 6

        if dataset == "results_medium_next_hour.csv" or dataset == '2':
            colnames = ['RowID', 'V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS',
                        'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3',
                        'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                        '1H_RS_IND']

            dense = 30
            modelname = "model_nextH"
            result = pd.read_csv("results_medium_next_hour.csv", names=colnames, sep=',', header=1)
            result.drop('RowID', 1, inplace=True)

            XY = result.values
            sel = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
            idx_y = -1

        epochs = input("How many epochs do you want to execute? ")
        while not epochs.isdigit():
            epochs = input("this is no valid value... please type numbers ")

        epochs = int(epochs)
        decide = 'yes'
        if epochs > 50:
            decide = input("Warning: this amount of epochs will take some time... are you sure?")

        if decide == '0' or decide == 'no':
            epochs = input("How many epochs do you want to execute? ")
            while not epochs.isdigit():
                epochs = input("this is no valid value... please type numbers ")
            epochs = int(epochs)

        if '1H_RS_IND' in result.head(0):
            RS_IND = result.get('1H_RS_IND').astype(str)
            result.drop('1H_RS_IND', 1, inplace=True)
            RS_IND = pd.DataFrame(RS_IND, columns=['1H_RS_IND'])
            result = result.join(RS_IND)

        # run neural network
        model = neural_net.nn(modelname, dense, result, epochs, sel=sel, idx_y=idx_y, newmodel=new_model)

        if '1H_RS_IND' in result.head(0):
            # test neural network on special data
            testData = pd.read_csv('testData.csv', names=colnames, sep=',', header=1)
            testData.drop('RowID', 1, inplace=True)

            XY = testData.values

            sel = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
            X, y = XY[:, sel], XY[:, -1]

            t.Tester.test_model(model, X, y)
    else:
        print("run Decision Tree with the dataset: results_medium_next_hour.csv")
        colnames = ['V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS',
                    'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3',
                    'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2', '3', '4', '5', '6', '7', '8', '1H_RS_IND']

        result = pd.read_csv('results_medium_next_hour.csv', names=colnames, sep=',', header=1)

        load = input("Does a trained model already exist and you just want to test it?")

        if load == '1' or load == 'yes':
            model =  pc.load(open('modeldt_1H.bin', 'rb'))

        else:
            model = tree.DecisionTree.run_decisionTree(result)

        t.Tester.test_dt(model,colnames)
else:
    print("The data you want to predict must be placed int the second row of testData.csv")
    print("The model needs to be created before you can take predictions!")

    nn = input("Okay. Do you want to get the prediction from the neural network(1/y(yes)? (From decision tree otherwise) ")

    colnames = ['RowID','V_N_x', 'V_S1_HHS', 'V_S1_NS', 'V_S2_HHS',
                'V_S2_NS', 'R1', 'RS_IND', 'TT_TU', 'RF_TU', 'SD_SO', 'F', 'D', 'sel_0', 'sel_1', 'sel_2', 'sel_3',
                'sel_4', 'sel_5', 'sel_6', 'sel_7', 'sel_8', '0', '1', '2', '3', '4', '5', '6', '7', '8', '1H_RS_IND']

    testData = pd.read_csv('testData.csv', names=colnames, sep=',', header=1)
    testData.drop('RowID', 1, inplace=True)

    XY = testData.values

    sel = [x for x in range(XY.shape[1]) if x != XY.shape[1] - 1]
    X, y = XY[:, sel], XY[:, -1]

    if nn == '1' or nn == 'yes' or nn == 'y':
        print("Make predictions with the neural net...")
        dense = 30
        modelname = "model_nextH"
        model = neural_net.load_model(False,modelname,dense)
        y_pred = model.predict(X)

        for y in y_pred:
            if y < 0.5:
                print("The chance that it will rain is low.")
            else:
                print("It's likely that it will rain in the next hour.")

    else:
        print("Make predictions with the Decision tree...")
        model =  pc.load(open('modeldt_1H.bin', 'rb'))
        y_pred = model.predict(X)
        for pred in y_pred:
            if pred < 0.5:
                print("The chance that it will rain is low.")
            else:
                print("It's likely that it will rain in the next hour.")
