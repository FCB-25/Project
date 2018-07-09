from keras.models import Sequential
from keras.layers import Dense
import keras.layers as layers
from sklearn.model_selection import train_test_split
from keras.models import model_from_json



class neural_net:
    def load_model(new_model, name, inputsize):
        if (new_model):
            # create model
            model = Sequential()
            model.add(Dense(512, input_dim=inputsize, activation='sigmoid'))
            model.add(Dense(256, activation='sigmoid'))
            model.add(Dense(29, activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
        else:
            # load json and create model
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(name)  # for normal NN "model.h5"
            print("Loaded model from disk")

        return model

    def saveModel(name, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(name)
        print("Saved model to disk")

    def nn(modelname, inputsize, result, epochs,idx_y,sel):
        """ load and prepare Data """

        XY = result.values

        X, y = XY[:, sel], XY[:, idx_y]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=0)

        model = neural_net.load_model(False, modelname, inputsize)

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

        # evaluate the model
        scores = model.evaluate(X_test, y_test)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        neural_net.saveModel(modelname, model)
        return model
