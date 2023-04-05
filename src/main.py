import logging as log
import os
import sys

import numpy as np
import tensorflow as tf
import tqdm as tqdm
from keras.layers import LSTM

import src.parameters as pm
from src.data import obtain_vectors
from src.log import initialize_log
import matplotlib.pyplot as plt


def obtain_model() -> tf.keras.Sequential:
    opt = tf.keras.optimizers.Adagrad(
        learning_rate=pm.LEARNING_RATE,
    )

    model = tf.keras.Sequential()
    model.add(
        LSTM(pm.UNITS, return_sequences=True, input_shape=(pm.STEPS_IN, pm.N_FEATURES))
    )
    model.add(LSTM(pm.UNITS))
    model.add(tf.keras.layers.Dense(pm.STEPS_OUT))
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy', 'mse'])
    return model


def predict(model):
    xx2, y2, scaler = obtain_vectors(pm.TEST_DATA, "test")

    __min = min(len(y2), pm.TEST_SIZE)
    xx2 = xx2[:__min + 1]
    y2 = y2[:__min + 1]

    yhat_history = []

    log.info("Predicting...")
    for i in tqdm.tqdm(range(__min)):
        x_input = np.array(xx2[i])
        x_input = x_input.reshape((1, pm.STEPS_IN, pm.N_FEATURES))
        yhat = model.predict(x_input, verbose=0)
        yhat_history.append(yhat)

    yhat_history = np.array(yhat_history)

    # Dump the prediction to a file
    os.makedirs(pm.LOG_FOLDER + "/pred", exist_ok=True)
    np.savetxt(pm.LOG_FOLDER + "/pred/prediction.csv", yhat_history[:, 0, 0], delimiter=",")
    np.savetxt(pm.LOG_FOLDER + "/pred/lower.csv", yhat_history[:, 0, 1], delimiter=",")
    np.savetxt(pm.LOG_FOLDER + "/pred/upper.csv", yhat_history[:, 0, 2], delimiter=",")
    np.savetxt(pm.LOG_FOLDER + "/pred/actual.csv", y2[:, 1], delimiter=",")

    # New plot
    plt.figure(figsize=(20, 10))
    plt.plot(yhat_history[:, 0, 0], label='target')
    plt.plot(yhat_history[:, 0, 1], label='lower')
    plt.plot(yhat_history[:, 0, 2], label='upper')
    plt.plot(y2[:, 1], label='actual')
    plt.legend()
    plt.savefig(pm.LOG_FOLDER + "/prediction.png")


def print_history(history):
    # list all data in history
    log.info("Available keys: " + str(history.history.keys()))
    try:
        # summarize history for accuracy
        plt.figure(figsize=(20, 10))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(pm.LOG_FOLDER + "/accuracy.png")
    except KeyError:
        pass
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(pm.LOG_FOLDER + "/loss.png")


def main():
    os.makedirs(pm.LOG_FOLDER, exist_ok=True)
    os.makedirs(pm.MODEL_DIR, exist_ok=True)
    os.makedirs(pm.DATA_DIR, exist_ok=True)

    initialize_log("INFO")
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

    log.info(f"Model name: {name}; Retrain: {retrain}; New model: {new_model}")


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
