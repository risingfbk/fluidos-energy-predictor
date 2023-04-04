import os

import numpy as np
import tensorflow as tf
import tqdm as tqdm
from keras.layers import LSTM

import src.parameters as pm
from src.data import obtain_vectors


def obtain_model() -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(
        LSTM(pm.UNITS, return_sequences=True, input_shape=(pm.STEPS_IN, pm.N_FEATURES))
    )
    model.add(LSTM(pm.UNITS))
    model.add(tf.keras.layers.Dense(pm.STEPS_OUT))
    model.compile(optimizer='adam', loss='mse')
    return model


def predict(model):
    xx2, y2, scaler = obtain_vectors(pm.TEST_FILENAME, "test")

    __min = min(len(y2), pm.TEST_SIZE)
    xx2 = xx2[:__min + 1]
    y2 = y2[:__min + 1]

    yhat_history = []

    print("Predicting...")
    for i in tqdm.tqdm(range(__min)):
        x_input = np.array(xx2[i])
        x_input = x_input.reshape((1, pm.STEPS_IN, pm.N_FEATURES))
        yhat = model.predict(x_input, verbose=0)
        yhat_history.append(yhat)

    yhat_history = np.array(yhat_history)
    import matplotlib.pyplot as plt

    plt.plot(yhat_history[:, 0, 0], label='target')
    plt.plot(yhat_history[:, 0, 1], label='lower')
    plt.plot(yhat_history[:, 0, 2], label='upper')
    plt.plot(y2[:, 1], label='actual')
    plt.legend()
    plt.show()


def main():
    new_model = False
    while True:
        name = input(f"Model name: [{pm.DEFAULT_MODEL}] ")
        if name == "":
            name = pm.DEFAULT_MODEL
        if os.path.exists(pm.MODEL_DIR + "/" + name + ".h5"):
            break
        else:
            print("Model not found. Would you like to train it as a new model?")
            while True:
                response = input("y/n: ")
                if response == "y":
                    new_model = True
                    break
                elif response == "n":
                    print("You supply a non-existent model and then refuse to train a new one. Goodbye.")
                    exit(1)
                else:
                    print("Invalid input")
                    continue
            break

    while True:
        if new_model:
            retrain = "t"
            break
        else:
            retrain = input("Retrain? [y/n]: ")
            if retrain not in ["y", "n"]:
                print("Invalid input")
                continue
            break

    # Load model eventually
    if not new_model:
        model = tf.keras.models.load_model(pm.MODEL_DIR + "/" + name + ".h5")
    else:
        model = obtain_model()

    print(model.summary())

    # Train
    if retrain == "y" or retrain == "t":
        xx, y, scaler = obtain_vectors(pm.TRAIN_DATA, "train")

        epochs = int(input("Epochs: "))
        history = model.fit(xx, y, epochs=epochs, verbose=1, validation_split=pm.SPLIT)
        tf.keras.models.save_model(model, pm.MODEL_DIR + "/" + name + ".h5")
        print(history)
        print("Saved model to disk")

    # Predict
    predict(model)


if __name__ == '__main__':
    main()
