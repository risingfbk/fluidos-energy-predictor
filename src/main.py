import os
import sys

import numpy as np
import tensorflow as tf
import tqdm as tqdm
from keras.layers import LSTM

import src.parameters as pm
from src.data import obtain_vectors
import logging as log

def obtain_model() -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(
        LSTM(pm.UNITS, return_sequences=True, input_shape=(pm.STEPS_IN, pm.N_FEATURES))
    )
    model.add(LSTM(pm.UNITS))
    model.add(tf.keras.layers.Dense(pm.STEPS_OUT))
    model.compile(optimizer='RMSprop', loss='mse', metrics=['accuracy', 'mse'])
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


def print_history(history):
    import matplotlib.pyplot as plt
    # list all data in history
    print(history.history.keys())
    try:
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    except KeyError:
        pass
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    log.basicConfig(level=log.INFO, stream=sys.stdout)
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
        log.info(f"Training data shape: {xx.shape} -> {y.shape}")

        try:
            epochs = int(input("Epochs: "))
        except ValueError:
            epochs = 0

        if epochs <= 0:
            log.error("Ah yes, training for 0 epochs. That's a good idea.")

        log.info(f"Training model for {epochs} epochs")
        history = model.fit(xx, y, epochs=epochs, verbose=1, validation_split=pm.SPLIT, shuffle=True)
        tf.keras.models.save_model(model, pm.MODEL_DIR + "/" + name + ".h5")
        print_history(history)
        log.info("Saved model to disk")

    # Predict
    predict(model)


if __name__ == '__main__':
    main()
