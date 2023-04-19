import logging as log

import numpy as np
from matplotlib import pyplot as plt

from src import parameters as pm
from src.log import initialize_log


def plot_prediction(yhat_history, truth, start=0, end=None):
    if end is None:
        end = len(yhat_history)

    plt.figure(figsize=(20, 10))
    plt.plot(yhat_history[start:end, 0, 0], label='prediction')
    plt.plot(yhat_history[start:end, 0, 1], label='lower')
    plt.plot(yhat_history[start:end, 0, 2], label='upper')
    plt.plot(truth[:, 1], label='actual')
    plt.legend()
    plt.savefig(pm.LOG_FOLDER + "/prediction.png")


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
    file="out/20230419_102508.180680__mf-mbp.fbkeduroam.it"
    yhat_history = np.load(file + "/pred/yhat_history.npy")
    y2 = np.load(file + "/pred/y2.npy")

    initialize_log("INFO", "plot")
    plot_prediction(yhat_history, y2, 0, 500)