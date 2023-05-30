import logging as log
import os
import random
from datetime import datetime

import tensorflow as tf

import src.model as modelmd
import src.parameters as pm
import src.secret_parameters as pms
from src.data import obtain_vectors, fetch_datasets
from src.support.log import initialize_log
from src.plot import plot_history


def main():
    initialize_log("INFO")

    # Block GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    os.makedirs(pm.LOG_FOLDER, exist_ok=True)
    os.makedirs(pm.MODEL_FOLDER, exist_ok=True)
    os.makedirs(pm.GCD_FOLDER, exist_ok=True)
    os.makedirs(pm.CACHE_FOLDER, exist_ok=True)

    train_data, test_data = fetch_datasets()

    new_model = False
    models = sorted([
        i.split(".")[0]
        for i in os.listdir(pm.MODEL_FOLDER)
    ])
    while True:
        print("---")
        if len(models) > 0:
            print(f"Available models:")
            for i in range(len(models)):
                model_name = models[i]
                if os.path.isfile(pm.MODEL_FOLDER + "/" + model_name + "/model.h5"):
                    print(f"\t{i + 1}) {model_name} (trained)")
                else:
                    print(f"\t{i + 1}) {model_name} (status unknown)")
            print()
        name = input(f"Model name:"
                     f"\n\t- '' or 'default' for default model ({pm.DEFAULT_MODEL})"
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

        decision = "n"

        print(f"Chosen model name: {name}")
        if os.path.exists(pm.MODEL_FOLDER + "/" + name + "/model.h5"):
            print("Model found. What would you like to do? [r]etrain / [c]ontinue / [e]xit")
            while True:
                response = input("r/c/e: ")
                match response:
                    case "r":
                        decision = "t"
                        break
                    case "c":
                        decision = "n"
                        break
                    case "e":
                        print("Exiting...")
                        exit(0)
                    case _:
                        print("Invalid input")
                        continue
            break
        else:
            print("Model not found. What would you like to do? [t]rain / [s]earch hyperparameters / [e]xit")
            while True:
                response = input("t/s/e: ")
                match response:
                    case "t":
                        decision = "t"
                        new_model = True
                        break
                    case "s":
                        from hp import search_params
                        search_params.search_hp(train_data)
                        exit(0)
                    case "e":
                        print("Exiting...")
                        exit(0)
                    case _:
                        print("Invalid input")
                        continue
            break

    if decision not in ["t", "n"]:
        raise EnvironmentError("Something went wrong. Decision is not t or n.")

    log.info(f"Model name: {name}; Retrain: {decision}; New model: {new_model}")
    pm.MODEL_FOLDER = os.path.join(pm.MODEL_FOLDER, name)
    os.makedirs(pm.MODEL_FOLDER, exist_ok=True)

    # Load model eventually
    if not new_model:
        model = tf.keras.models.load_model(pm.MODEL_FOLDER + "/model.h5")
    else:
        model = modelmd.new_model()

    print(model.summary())

    # Train
    if decision == "t":
        epochs = 0
        while True:
            try:
                epochs = input("Epochs: ")
                epochs = float(epochs)

                if int(epochs) != epochs:
                    print("Ah yes, training for a non-integer amount of epochs. That's a good idea.")
                    continue

                epochs = int(epochs)

                if epochs <= 0:
                    print("Ah yes, training for 0 epochs. That's a good idea. Congratulations, you broke math.")
                    continue

                break
            except ValueError:
                print("Ah yes, training for NaN epochs. That's a good idea.")
                continue

        xx, y = obtain_vectors(train_data)

        log.info(f"Training data shape: {xx.shape} -> {y.shape}")

        cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pm.PATIENCE),
              # tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.0001),
              tf.keras.callbacks.BackupAndRestore(pm.MODEL_FOLDER + "/backup"),
              tf.keras.callbacks.ModelCheckpoint(pm.MODEL_FOLDER + "/model.h5",
                                                 save_best_only=True, save_weights_only=False,
                                                 monitor='val_loss', mode='min')]

        log.info(f"Training model for {epochs} epochs")
        history = model.fit(xx, y, epochs=epochs, verbose=1, validation_split=pm.SPLIT, shuffle=True,
                            callbacks=cb)

        tf.keras.models.save_model(model, pm.MODEL_FOLDER + "/last_model.h5")

        log.info("Saved model to disk")
        plot_history(history)

        # restore best model
        model = tf.keras.models.load_model(pm.MODEL_FOLDER + "/model.h5")

    # Predict
    results = modelmd.predict(model, test_data)
    log.info(f"Predictions: {results}")


if __name__ == '__main__':
    main()
