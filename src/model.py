import numpy as np
import tensorflow as tf
from keras.layers import LSTM
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
    #loss = tf.keras.losses.MeanSquaredError()

    accuracy = [tf.keras.metrics.MeanSquaredError()]

    model = tf.keras.Sequential()
    model.add(LSTM(pm.STEPS_OUT, input_shape=(pm.STEPS_IN, pm.N_FEATURES)))
    # model.add(LSTM(pm.UNITS, return_sequences=True, input_shape=(pm.STEPS_IN, pm.N_FEATURES)))
    # model.add(LSTM(pm.UNITS))
    # model.add(tf.keras.layers.Dropout(pm.DROPOUT))
    # model.add(tf.keras.layers.Dense(pm.STEPS_OUT, kernel_initializer='normal', activation='relu'))
    # model.add(tf.keras.layers.Dense(pm.STEPS_OUT))

    model.compile(optimizer=opt, loss=custom_loss, metrics=[accuracy])
    model.build(input_shape=(pm.STEPS_IN, pm.N_FEATURES))

    return model


def predict(model: tf.keras.Sequential, test_data: list[str]):
    xx2, y2, scaler = obtain_vectors(test_data[0], "test", keep_scaler=True)

    __min = min(len(y2), pm.TEST_SIZE)
    xx2 = xx2[:__min]
    y2 = y2[:__min]

    y2_history = []
    yhat_history = []
    columns = []

    # predict every pm.STEPS_OUT steps
    for i in tqdm_wrapper(range(0, __min, pm.STEPS_OUT)):
        x_input = np.array(xx2[i])
        x_input = x_input.reshape((1, pm.STEPS_IN, pm.N_FEATURES))
        y2_input = np.array(y2[i])
        yhat = model.predict(x_input, verbose=0)
        yhat_history.append(yhat)
        y2_history.append(y2_input)
        columns.append(i)

    yhat_history = np.array(yhat_history).flatten()
    y2_history = np.array(y2_history).flatten()

    save_prediction(yhat_history, y2_history)
    plot_prediction(test_data, yhat_history, y2_history, columns)

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
        if abs(yhat_history[i] - y2_history[i]) <= std:
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
