from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Activation
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

class neural_net:
    def load_model(new_model, name, inputsize,train_X):
        if (new_model):
            # create model
            model = Sequential()
            model.add(LSTM(128,input_shape=(train_X.shape[1], train_X.shape[2]),activation='sigmoid'))
            model.add(Dense(inputsize))
            model.add(Dense(1))
            model.add(Activation('tanh'))
        else:
            # load json and create model
            json_file = open(name + ".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(name + ".h5")  # for normal NN "model.h5"
            print("Loaded model from disk")
        return model

    def saveModel(name, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(name + ".h5")
        print("Saved model to disk")

    def nn(modelname, inputsize, result, epochs, idx_y, sel, newmodel):
        """ load and prepare Data """
        scaler = StandardScaler()
        XY = result.values

        X, y = XY[:, sel], XY[:, idx_y]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=0)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        X_val = scaler.fit_transform(X_val)

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        model = neural_net.load_model(newmodel, modelname, inputsize,X_train)

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse','accuracy'])

        # Fit the model
        model.fit(X_train,y_train,epochs=epochs, batch_size=72, validation_data=(X_val, y_val), verbose=2, shuffle=False)

        # evaluate the model
        scores = model.evaluate(X_test, y_test)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))

        neural_net.saveModel(modelname, model)
        return model
