import logging as log
import os

import numpy as np
from matplotlib import pyplot as plt

from src import parameters as pm
from src import secrets as pms
from src.log import initialize_log


def save_prediction(yhat_history, y2):
    # Dump the prediction to a file
    os.makedirs(pm.LOG_FOLDER + "/pred", exist_ok=True)
    np.savetxt(pm.LOG_FOLDER + "/pred/prediction.csv", yhat_history[:, 0, 0], delimiter=",")
    np.savetxt(pm.LOG_FOLDER + "/pred/lower.csv", yhat_history[:, 0, 1], delimiter=",")
    np.savetxt(pm.LOG_FOLDER + "/pred/upper.csv", yhat_history[:, 0, 2], delimiter=",")
    np.savetxt(pm.LOG_FOLDER + "/pred/actual.csv", y2[:, 1], delimiter=",")
    np.save(pm.LOG_FOLDER + "/pred/yhat_history.npy", yhat_history)
    np.save(pm.LOG_FOLDER + "/pred/y2.npy", y2)

def plot_prediction(yhat_history, truth, start=0, end=None):
    if end is None:
        end = len(yhat_history)

    plt.figure(figsize=(20, 10))
    plt.plot(yhat_history[start:end, 0, 0], label='prediction')
    plt.plot(yhat_history[start:end, 0, 1], label='lower')
    plt.plot(yhat_history[start:end, 0, 2], label='upper')
    plt.plot(truth[start:end, 1], label='actual')
    plt.legend()
    plt.savefig(pm.LOG_FOLDER + f"/prediction-{start}-{end}.png")

def plot_history(history):
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


if __name__ == "__main__":
    file = pms.PLOT_FILE
    yhat_history = np.load(file + "/pred/yhat_history.npy")
    y2 = np.load(file + "/pred/y2.npy")

    initialize_log("INFO", "plot")
    for i in range(0, len(yhat_history), 500):
        plot_prediction(yhat_history, y2, i, i + 500)

    print("Done!")
