import logging as log
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import LSTM

import src.parameters as pm
from src.data import obtain_vectors
from src.log import initialize_log, tqdm_wrapper


def obtain_model() -> tf.keras.Sequential:
    # opt = tf.keras.optimizers.Adagrad(learning_rate=pm.LEARNING_RATE)
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=pm.LEARNING_RATE)
    loss = tf.keras.losses.MeanSquaredError()

    model = tf.keras.Sequential()
    model.add(
        LSTM(pm.UNITS, return_sequences=True, input_shape=(pm.STEPS_IN, pm.N_FEATURES))
    )
    model.add(LSTM(pm.UNITS))
    model.add(tf.keras.layers.Dense(pm.STEPS_OUT))

    model.compile(optimizer=opt, metrics=['accuracy', 'mse'], loss=loss)

    return model


def predict(model: tf.keras.Model, test_data: list[str]):
    xx2, y2 = obtain_vectors(test_data, "test")

    __min = min(len(y2), pm.TEST_SIZE)
    xx2 = xx2[:__min + 1]
    y2 = y2[:__min + 1]

    yhat_history = []

    log.info("Predicting...")
    for i in tqdm_wrapper(range(__min)):
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

    # New plot. We plot y as a line, while for the predictions,
    # each data point is an estimate for the subsequent pm.YWINDOW data points.
    # We plot the lower and upper bounds as a shaded area.

    #     plt.figure(figsize=(20, 10))
    #     plt.plot(yhat_history[:, 0, 0], label='target')
    #     plt.plot(y2[:, 1], label='actual')
    #     plt.fill_between(
    #         range(len(yhat_history[:, 0, 0])),
    #         yhat_history[:, 0, 1],
    #         yhat_history[:, 0, 2],
    #         alpha=0.5,
    #         label='prediction interval'
    #     )
    #     plt.legend()
    #     plt.savefig(pm.LOG_FOLDER + "/prediction.png")

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
    initialize_log("INFO")

    os.makedirs(pm.LOG_FOLDER, exist_ok=True)
    os.makedirs(pm.MODEL_DIR, exist_ok=True)
    os.makedirs(pm.DATA_DIR, exist_ok=True)

    files_to_be_chosen = pm.TRAIN_FILE_AMOUNT + pm.TEST_FILE_AMOUNT
    files = os.listdir(pm.DATA_DIR)
    if len(files) < files_to_be_chosen:
        raise EnvironmentError(f"Insufficient data. You need at least {files_to_be_chosen} files in {pm.DATA_DIR}")

    # Choose files to be used for training and testing
    chosen = random.sample(files, files_to_be_chosen)
    train_data = chosen[:pm.TRAIN_FILE_AMOUNT]
    test_data = chosen[pm.TRAIN_FILE_AMOUNT:]

    log.info("Training files: " + str(train_data))
    log.info("Testing files: " + str(test_data))

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
        xx, y = obtain_vectors(train_data, "train")
        log.info(f"Training data shape: {xx.shape} -> {y.shape}")

        try:
            epochs = int(input("Epochs: "))
        except ValueError:
            epochs = 0

        if epochs <= 0:
            log.error("Ah yes, training for 0 epochs. That's a good idea.")

        log.info(f"Training model for {epochs} epochs")
        history = model.fit(xx, y, epochs=epochs, verbose=1, validation_split=pm.SPLIT, shuffle=True,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pm.PATIENCE)])
        tf.keras.models.save_model(model, pm.MODEL_DIR + "/" + name + ".h5")
        print_history(history)
        log.info("Saved model to disk")
    # Predict
    predict(model, test_data)


if __name__ == '__main__':
    main()
