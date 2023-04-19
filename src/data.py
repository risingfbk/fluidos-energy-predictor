import logging as log
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src import parameters as pm
from src.log import tqdm_wrapper


def load_data(file: str, mode: str) -> tuple[None, np.ndarray]:
    threshold = pm.TRAIN_SIZE if mode == "train" else pm.TEST_SIZE

    res = []

    file = pm.DATA_DIR + "/" + file

    if os.path.exists(file + ".npy"):
        log.info(f"Reading cached data from {file}.npy")
        return None, np.load(file + ".npy")  # np.load(file + "_ts.npy")

    log.info(f"Reading data from {file}")
    with open(file, 'r') as f:
        # data be like
        # 38154,69b2e9507bc26035b4f7d2ce084d17029c40ddf06a7b846f667a287896c54e4e,
        # 47f4004bd44f0fa53cd04c14d073518dde281f56c8244de8f81248490a0fbd48,
        # K8adcea9c2aca0eebe4ebbead8ad711caf677287d58b396389a6b16c2bf09ae06,
        # 0.05243749999984478,0.3073234558105469,300000
        # that is, timestamp,container_id,task_id,node_id,cpu,mem,?
        # we are only interested in timestamp,cpu
        for line in tqdm_wrapper(f):
            fts, _, _, _, cpu, *_ = line.split(',')
            # timestamp be like: filename:12341513
            filename, timestamp = fts.split(':')

            try:
                res.append((int(timestamp), float(cpu)))
            except ValueError:
                pass
            if len(res) > threshold + 1:
                break

    res.sort(key=lambda x: x[0])
    float_inputs = [x[1] for x in res]

    log.debug("Computing moving average...")
    for _ in tqdm_wrapper(range(pm.MVAVG)):
        for i in range(1, len(float_inputs) - 1):
            float_inputs[i] = (float_inputs[i] + float_inputs[i - 1]) / 2

    # float_inputs = float_inputs[MVAVG:]
    # ts = ts[MVAVG:]

    float_inputs = np.array(float_inputs)
    np.save(file + ".npy", float_inputs)
    # np.save(file + "_ts.npy", np.array(ts))

    return None, float_inputs


def trans_foward(scaler: MinMaxScaler, arr):
    out_arr = scaler.transform(arr.reshape(-1, 1))
    return out_arr.flatten()


def trans_back(scaler: MinMaxScaler, arr):
    out_arr = scaler.inverse_transform(arr.flatten().reshape(-1, 1))
    return out_arr.flatten()


def split_sequence(sequence, n_steps_in, n_steps_out, ywindow, filename):
    xx, y = [], []
    seq_filename = pm.DATA_DIR + '/' + \
                   'samples_' + ",".join([str(a) for a in [n_steps_in, n_steps_out, ywindow, filename, len(sequence)]])

    if os.path.exists(seq_filename + '.npy'):
        log.info(f"Using cached sequences for {seq_filename}...")
        return np.load(seq_filename + '.npy'), np.load(seq_filename + '_y.npy')

    else:
        log.info(f"Generating sequences for {seq_filename}...")
        for i in tqdm_wrapper(range(len(sequence) - ywindow - n_steps_in + 1)):
            # find the end of this pattern
            end_ix = i + n_steps_in

            # gather input and output parts of the pattern
            # print(sequence[end_ix:end_ix+ywindow])
            seq_x, seq_y = sequence[i:end_ix], \
                [np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_TARGET),
                 np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_LOWER),
                 np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_UPPER)]
            xx.append(seq_x)
            y.append(seq_y)

        np.save(seq_filename + '.npy', np.array(xx))
        np.save(seq_filename + '_y.npy', np.array(y))
        # print(np.array(X), np.array(y))
        return np.array(xx), np.array(y)


def obtain_vectors(data_file: str | list[str], mode: str) -> (np.ndarray, np.ndarray):  # , MinMaxScaler):
    if isinstance(data_file, list):
        xxmerge, ymerge = [], []
        for i in range(len(data_file)):
            # xx, y, _ = obtain_vectors(data_file[i], mode)
            xx, y = obtain_vectors(data_file[i], mode)
            xxmerge.append(xx)
            ymerge.append(y)

        xx = np.concatenate(xxmerge)
        y = np.concatenate(ymerge)
        return xx, y  # , scaler

    ts, float_inputs = load_data(data_file, mode)
    float_inputs = np.array(float_inputs)

    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    # noinspection PyTypeChecker
    scaler: MinMaxScaler = scaler.fit(np.array(float_inputs).reshape(-1, 1))
    dataset = trans_back(scaler, float_inputs)

    # split into samples
    xx, y = split_sequence(dataset, pm.STEPS_IN, pm.STEPS_OUT, pm.YWINDOW, data_file)
    xx = xx.reshape((xx.shape[0], xx.shape[1], pm.N_FEATURES))
    log.debug("Working with", xx.shape, " ", y.shape, "samples")

    return xx, y  # , scaler
