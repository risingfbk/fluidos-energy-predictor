import logging as log

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

from src import parameters as pm
from src.data import obtain_vectors
from src.log import tqdm_wrapper
from src.plot import save_prediction, plot_prediction


def custom_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


def new_model() -> tf.keras.Sequential:
    # opt = tf.keras.optimizers.Adagrad(learning_rate=pm.LEARNING_RATE)
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=pm.LEARNING_RATE)
    custom_loss = tf.keras.losses.MeanSquaredError()

    accuracy = [tf.keras.metrics.MeanSquaredError()]

    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.LSTM(pm.STEPS_OUT, input_shape=(pm.STEPS_IN, pm.N_FEATURES)))
    # model.add(tf.keras.layers.Dense(pm.STEPS_OUT, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.LSTM(pm.UNITS, return_sequences=True, input_shape=(pm.STEPS_IN, pm.N_FEATURES)))
    model.add(tf.keras.layers.LSTM(pm.UNITS))
    # model.add(tf.keras.layers.Dropout(pm.DROPOUT))
    # model.add(tf.keras.layers.Dense(pm.STEPS_OUT, kernel_initializer='normal', activation='relu'))
    # output layer outputs next pm.STEPS_OUT values for CPU and memory, so shape is (pm.STEPS_OUT, 2)
    model.add(tf.keras.layers.Dense(pm.N_FEATURES * pm.STEPS_OUT, kernel_initializer='normal', activation='relu'))

    model.compile(optimizer=opt, loss=custom_loss, metrics=[accuracy])
    model.build(input_shape=(pm.STEPS_IN, pm.N_FEATURES))

    return model


def predict(model: tf.keras.Sequential, test_data: list[str]):
    if len(test_data) > 1:
        log.error("Only one file is supported for prediction, using the first one")
    xx2, y2 = obtain_vectors(test_data[0])
    y2 = y2.reshape((len(y2), pm.STEPS_OUT, pm.N_FEATURES))

    __min = min(len(y2), pm.TEST_SIZE)
    xx2 = xx2[:__min]
    y2 = y2[:__min]

    yhat_history = np.ndarray(shape=(0, pm.STEPS_OUT, pm.N_FEATURES))
    columns = []
    y2_history = np.ndarray(shape=(0, pm.STEPS_OUT, pm.N_FEATURES))

    # predict every pm.STEPS_OUT steps
    for i in tqdm_wrapper(range(0, __min, pm.STEPS_OUT)):
        x_input = xx2[i]
        x_input = x_input.reshape((1, pm.STEPS_IN, pm.N_FEATURES))
        y2_input = y2[i]
        y2_input = y2_input.reshape((1, pm.STEPS_OUT, pm.N_FEATURES))

        yhat = model.predict(x_input, verbose=0)
        # model predict returns a flat vector, so reshape it to (pm.STEPS_OUT, pm.N_FEATURES)
        yhat = yhat.reshape((pm.STEPS_OUT, pm.N_FEATURES))
        yhat_history = np.append(yhat_history, [yhat], axis=0)
        y2_history = np.append(y2_history, y2_input, axis=0)

        columns.append(i)

    plot_prediction(test_data, yhat_history, y2_history, columns)

    # reshape again to compute metrics and save the prediction
    yhat_history = yhat_history.reshape((len(yhat_history), pm.STEPS_OUT * pm.N_FEATURES))
    y2_history = y2_history.reshape((len(y2_history), pm.STEPS_OUT * pm.N_FEATURES))

    save_prediction(yhat_history, y2_history)

    r2 = r2_score(y2_history, yhat_history)
    mse = mean_squared_error(y2_history, yhat_history)
    # Calculate standard deviation as a measure of accuracy
    # for each data point, the prediction is within 1 std of the actual value
    std = np.std(yhat_history - y2_history)

    # integrate both y2 and yhat_history to get the total workload
    y2_sum = np.cumsum(y2_history)[len(y2_history) - 1]
    yhat_sum = np.cumsum(yhat_history)[len(yhat_history) - 1]

    yes = 0
    no = 0
    # for each sample, the sample is correct if the prediction is within 1 std of the actual value
    for i in range(len(y2_history)):
        if abs(yhat_history[i] - y2_history[i]).all() <= std:
            yes += 1
        else:
            no += 1

    return {
        "r2": r2,
        "mse": mse,
        "approx": {
            "yes": yes,
            "no": no,
            "std": std,
            "prop": yes / (yes + no),
        },
        "area": {
            "truth": y2_sum,
            "pred": yhat_sum,
            "diff": abs(y2_sum - yhat_sum),
            "prop": abs(y2_sum - yhat_sum) / y2_sum,
        },
    }
