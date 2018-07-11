from keras.models import Sequential
from keras import metrics
from keras.layers import Dense, LSTM, TimeDistributed, Activation
from sklearn.model_selection import train_test_split
from keras.models import model_from_json


class neural_net:
    def load_model(new_model, name, inputsize):
        if (new_model):
            # create model
            model = Sequential()
            model.add(LSTM())
            model.add(Dense(512, input_dim=inputsize, activation='sigmoid'))
            model.add(Dense(256, activation='sigmoid'))
            model.add(Dense(inputsize, activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
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

        XY = result.values

        X, y = XY[:, sel], XY[:, idx_y]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=0)

        model = neural_net.load_model(newmodel, modelname, inputsize)

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.categorical_accuracy])

        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

        # evaluate the model
        scores = model.evaluate(X_test, y_test)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        neural_net.saveModel(modelname, model)
        return model

    def fit_lstm(train, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])

        model = Sequential()
        model.add(Dense(512, input_dim=30, activation='sigmoid'))
        model.add(LSTM(256, input_shape=(512, 30), return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(TimeDistributed(Dense(64)))
        model.add(Activation('softmax'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print("fit model...")
        model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=0, shuffle=False)
        return model
