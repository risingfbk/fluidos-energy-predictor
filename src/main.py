import logging as log
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras.layers import LSTM

import src.parameters as pm
import src.secret_parameters as pms
from src.data import obtain_vectors, trans_back
from src.log import initialize_log, tqdm_wrapper
from src.plot import plot_prediction, plot_history, save_prediction


def obtain_model() -> tf.keras.Sequential:
    # opt = tf.keras.optimizers.Adagrad(learning_rate=pm.LEARNING_RATE)
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=pm.LEARNING_RATE)
    loss = tf.keras.losses.MeanSquaredError()

    model = tf.keras.Sequential()
    model.add(LSTM(pm.UNITS, return_sequences=True, input_shape=(pm.STEPS_IN, pm.N_FEATURES)))
    model.add(LSTM(pm.UNITS))
    # model.add(tf.keras.layers.Dropout(pm.DROPOUT))
    # model.add(tf.keras.layers.Dense(pm.STEPS_OUT, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(pm.STEPS_OUT))

    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    return model


def predict(model: tf.keras.Sequential, test_data: list[str]):
    xx2, y2, scaler = obtain_vectors(test_data[0], "test", keep_scaler=True)

    __min = min(len(y2), pm.TEST_SIZE)
    xx2 = xx2[:__min]
    y2 = y2[:__min]

    yhat_history = []

    for i in tqdm_wrapper(range(__min)):
        x_input = np.array(xx2[i])
        x_input = x_input.reshape((1, pm.STEPS_IN, pm.N_FEATURES))
        yhat = model.predict(x_input, verbose=0)
        yhat = trans_back(scaler, yhat)
        yhat_history.append(yhat)

    yhat_history = np.array(yhat_history)
    save_prediction(yhat_history, y2)
    plot_prediction(yhat_history, y2)


def main():
    initialize_log("INFO")

    os.makedirs(pm.LOG_FOLDER, exist_ok=True)
    os.makedirs(pm.MODEL_DIR, exist_ok=True)
    os.makedirs(pm.DATA_DIR, exist_ok=True)

    banlist = []
    if os.path.exists("banlist"):
        with open("banlist", "r") as f:
            for line in f:
                banlist.append(line.strip() + ".csv")
    else:
        with open("banlist", "w") as f:
            f.write("")

    files_to_be_chosen = 0
    if pms.TEST_FILES is not None:
        log.info("Verifying test files...")
        test_data = pms.TEST_FILES
        for file in test_data:
            if file in banlist:
                raise EnvironmentError(f"File {file} is banned. Please refrain from using it ever again.")
            if not os.path.exists(os.path.join(pm.DATA_DIR, file)):
                raise EnvironmentError(f"File {file} does not exist.")
    else:
        log.info("No test files specified. Choosing randomly.")
        test_data = None
        files_to_be_chosen += pm.TEST_FILE_AMOUNT

    if pms.TRAIN_FILES is not None:
        log.info("Verifying train files...")
        train_data = pms.TRAIN_FILES
        for file in train_data:
            if file in banlist:
                raise EnvironmentError(f"File {file} is banned. Please refrain from using it ever again.")
            if not os.path.exists(os.path.join(pm.DATA_DIR, file)):
                raise EnvironmentError(f"File {file} does not exist.")
    else:
        log.info("No train files specified. Choosing randomly.")
        train_data = None
        files_to_be_chosen += pm.TRAIN_FILE_AMOUNT

    if files_to_be_chosen == 0:
        raise EnvironmentError("You need to specify at least one file for training and one for testing.")

    if train_data is None or test_data is None:
        files = os.listdir(pm.DATA_DIR)
        files = [x for x in files if x.endswith(".csv")]

        if len(files) < files_to_be_chosen:
            raise EnvironmentError(f"Insufficient data. You need at least {files_to_be_chosen} files in {pm.DATA_DIR}")

        # Choose files to be used for training and testing
        chosen = random.sample(files, files_to_be_chosen)
        if train_data is None:
            train_data = random.sample(chosen, pm.TRAIN_FILE_AMOUNT)
            chosen = [x for x in chosen if x not in train_data]

        if test_data is None:
            test_data = random.sample(chosen, pm.TEST_FILE_AMOUNT)
            chosen = [x for x in chosen if x not in test_data]

        if len(chosen) > 0:
            raise EnvironmentError(f"Something went wrong. {len(chosen)} files were not used.")

    if len(train_data) == 0 or len(test_data) == 0:
        raise EnvironmentError(
            "Something went wrong. You need to specify at least one file for training and one for testing.")

    for file in test_data + train_data:
        if file in banlist:
            raise EnvironmentError(f"File {file} is banned. Please refrain from using it ever again.")

    log.info("Training files: " + str(train_data))
    log.info("Testing files: " + str(test_data))

    new_model = False
    models = [i.split(".")[0] for i in os.listdir(pm.MODEL_DIR)]
    while True:
        print("---")
        if len(models) > 0:
            print(f"Available models:")
            for i in range(len(models)):
                print(f"\t{i + 1}) {models[i]}")
            print()
        name = input(f"Model name:"
                     f"\n\t- '' or 'default' for default model ({pm.DEFAULT_MODEL})" + \
                     f"\n\t- 'dt' for datetime"
                     f"\n\t- [name] for name" + \
                     (f"\n\t- [number] for a specific model in list" if len(models) > 0 else "") + \
                     f"\n > ")
        match name:
            case "" | "default":
                name = pm.DEFAULT_MODEL
            case "dt":
                name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            case _:
                try:
                    name = models[int(name) - 1]
                except:
                    pass

        print(f"Chosen model name: {name}")
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
                    print(
                        "You supply a non-existent model and then refuse to train a new one. Do you want an applause?")
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
        try:
            epochs = input("Epochs: ")
            epochs = float(epochs)
            if int(epochs) != epochs:
                log.error("Ah yes, training for a non-integer amount of epochs. That's a good idea.")
                exit(1)
            epochs = int(epochs)
        except ValueError:
            log.error("Ah yes, training for NaN epochs. That's a good idea.")
            exit(1)

        if epochs <= 0:
            log.error("Ah yes, training for 0 epochs. That's a good idea. Congratulations, you broke math.")
            exit(1)

        xx, y, _ = obtain_vectors(train_data, "train")

        log.info(f"Training data shape: {xx.shape} -> {y.shape}")

        log.info(f"Training model for {epochs} epochs")
        history = model.fit(xx, y, epochs=epochs, verbose=1, validation_split=pm.SPLIT, shuffle=True,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=pm.PATIENCE)])
        tf.keras.models.save_model(model, pm.MODEL_DIR + "/" + name + ".h5")
        log.info("Saved model to disk")
        plot_history(history)

    # Predict
    predict(model, test_data)


if __name__ == '__main__':
    main()
