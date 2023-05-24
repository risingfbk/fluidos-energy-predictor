import logging as log
import os

import numpy as np
from matplotlib import pyplot as plt

from src import parameters as pm
from src import secret_parameters as pms
from src.log import initialize_log


def save_prediction(yhat, y2):
    # Dump the prediction to a file
    os.makedirs(pm.LOG_FOLDER + "/pred", exist_ok=True)
    np.savetxt(pm.LOG_FOLDER + "/pred/prediction.csv", yhat, delimiter=",")
    # np.savetxt(pm.LOG_FOLDER + "/pred/lower.csv", yhat[:,  1], delimiter=",")
    # np.savetxt(pm.LOG_FOLDER + "/pred/upper.csv", yhat[:, 2], delimiter=",")
    np.savetxt(pm.LOG_FOLDER + "/pred/actual.csv", y2, delimiter=",")
    np.save(pm.LOG_FOLDER + "/pred/yhat_history.npy", yhat)
    np.save(pm.LOG_FOLDER + "/pred/y2.npy", y2)


def plot_prediction(test_data, yhat, truth, columns, start=0, end=None):
    if end is None:
        end = len(yhat)

    plt.figure(figsize=(20, 10))
    # First, plot what was before
    if test_data is not None:
        pass
    plt.plot(yhat, label='prediction', linestyle='-.', alpha=.7, color='r')
    # plt.plot(yhat[start:end,  1], label='lower', linestyle='--', alpha=.6, color='r')
    # plt.plot(yhat[start:end,  2], label='upper', linestyle='--', alpha=.6, color='r')
    plt.plot(truth, label='actual', linestyle='-', alpha=.5, color='b')
    # plt.plot(truth[start:end, 1], label='actual lower', linestyle=':', alpha=.4, color='b')
    # plt.plot(truth[start:end, 2], label='actual upper', linestyle=':', alpha=.4, color='b')
    for i in columns:
        plt.axvline(x=i, linestyle='--', alpha=.3, color='g')

    # fill with color
    # plt.fill_between(
    #     np.arange(start, end),
    #     yhat[start:end, 1],
    #     yhat[start:end,  2],
    #     color='r',
    #     alpha=.15
    # )
    #
    # plt.fill_between(
    #     np.arange(start, end),
    #     truth[start:end, 1],
    #     truth[start:end, 2],
    #     color='b',
    #     alpha=.15
    # )
    plt.legend()
    plt.savefig(pm.LOG_FOLDER + f"/prediction-{start}-{end}.png")


def plot_history(history):
    # list all data in history
    log.info("Available keys: " + str(history.history.keys()))

    for key in history.history.keys():
        if "val" not in key:
            continue
        plt.figure(figsize=(20, 10))
        plt.plot(history.history[key])
        plt.plot(history.history[key.replace("val_", "")])
        plt.title('model ' + key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(pm.LOG_FOLDER + "/" + key.replace("val_", "") + ".png")
        plt.close()

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


def plot_splitter():
    file = pms.PLOT_FILE
    history = np.load(file + "/pred/yhat_history.npy")
    truth = np.load(file + "/pred/y2.npy")

    initialize_log("INFO", "plot")
    for i in range(0, len(history), 500):
        plot_prediction(history, truth, i, i + 500)

    print("Done!")


if __name__ == "__main__":
    plot_splitter()
