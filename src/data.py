import logging as log
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src import parameters as pm
from src.log import tqdm_wrapper


def load_data(file: str, mode: str) -> tuple[None, np.ndarray]:
    res = []

    npyfile = os.path.join(pm.CACHE_DIR, file.split("/")[-1].replace(".csv", ".npy"))
    file = os.path.join(pm.DATA_DIR, file)

    if os.path.exists(npyfile):
        log.info(f"Reading cached data from {npyfile}")
        return None, np.load(npyfile)

    log.info(f"Reading data from {file}")
    with open(file, 'r') as f:
        # data be like
        # timestamp,container_id,task_id,node_id,cpu,mem,idk
        # 38154,69b..,47f4...,K8ad..., 0.052437,0.3073234,300000

        for line in tqdm_wrapper(f):
            if 'timestamp' in line:
                continue

            fts, _, _, _, cpu, *_ = line.split(',')

            # depending on the file, the timestamp can be in different positions
            # timestamp be like: filename:12341513
            if ":" in fts:
                _, timestamp = fts.split(':')
            else:
                timestamp = fts

            try:
                res.append((int(timestamp), float(cpu)))
            except ValueError as e:
                log.warning(f"Error while parsing {line}: {e}")
                pass
            # if len(res) > threshold + 1:
            #    break

    res.sort(key=lambda x: x[0])
    float_inputs = [x[1] for x in res]

    log.debug("Computing moving average...")
    # mv_avg_window = pm.SMOOTH_WINDOW
    # mv_avg_runs = pm.SMOOTH_RUNS

    # for _ in range(mv_avg_runs):
    #     for i in tqdm_wrapper(range(len(float_inputs) - mv_avg_window)):
    #        new_inputs.append(np.mean(float_inputs[i:i + mv_avg_window]))

    float_inputs = np.array(float_inputs)
    np.save(npyfile, float_inputs)
    # np.save(file + "_ts.npy", np.array(ts))

    return None, float_inputs


def trans_forward(scaler: MinMaxScaler, arr):
    out_arr = scaler.transform(arr.reshape(-1, 1))
    return out_arr.flatten()


def trans_back(scaler: MinMaxScaler, arr):
    out_arr = scaler.inverse_transform(arr.flatten().reshape(-1, 1))
    return out_arr.flatten()


def split_sequence(sequence, n_steps_in, n_steps_out, filename):
    xx, y = [], []

    simple_filename = filename.split("/")[-1].replace(".csv", "")
    desc = ",".join([str(a) for a in [n_steps_in, n_steps_out, simple_filename, len(sequence)]])

    seq_filename_npy = pm.CACHE_DIR + '/' + "samples_" + desc + ".npy"
    seq_filename_y_npy = pm.CACHE_DIR + '/' + "samples_" + desc + "_y.npy"
    seq_filename = pm.DATA_DIR + '/' + "samples_" + desc

    if os.path.exists(seq_filename_npy) and os.path.exists(seq_filename_y_npy):
        log.info(f"Using cached sequences for {seq_filename}...")
        return np.load(seq_filename_npy), np.load(seq_filename_y_npy)
    elif os.path.exists(seq_filename_npy) or os.path.exists(seq_filename_y_npy):
        raise Exception(f"Only one of {seq_filename_npy} or {seq_filename_y_npy} exists!")
    else:
        log.info(f"Generating sequences for {seq_filename}...")
        for i in tqdm_wrapper(range(len(sequence) - n_steps_out - n_steps_in + 1)):
            # find the end of this pattern
            end_ix = i + n_steps_in

            # gather input and output parts of the pattern
            # print(sequence[end_ix:end_ix+ywindow])
            seq_x = sequence[i:end_ix]
            seq_y = sequence[end_ix:end_ix + n_steps_out]
            # [np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_TARGET),
            #  np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_LOWER),
            #  np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_UPPER)]
            xx.append(seq_x)
            y.append(seq_y)

        np.save(seq_filename_npy, np.array(xx))
        np.save(seq_filename_y_npy, np.array(y))

        return np.array(xx), np.array(y)


def obtain_vectors(data_file: str | list[str], mode: str, keep_scaler: bool = False) -> (
        np.ndarray, np.ndarray, MinMaxScaler):
    if isinstance(data_file, list):
        if keep_scaler:
            raise ValueError("keep_scaler is not supported for multiple data files")

        if len(data_file) == 0:
            raise ValueError("Empty list of data files")

        xxmerge, ymerge = [], []
        for i in range(len(data_file)):
            __xx, __y, _ = obtain_vectors(data_file[i], mode)
            xxmerge.append(__xx)
            ymerge.append(__y)

        xx = np.concatenate(xxmerge)
        y = np.concatenate(ymerge)

        return xx, y, None

    ts, float_inputs = load_data(data_file, mode)
    dataset = np.array(float_inputs)

    # split into samples
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(dataset.reshape(-1, 1))
    dataset = trans_forward(scaler, dataset)

    xx, y = split_sequence(dataset, pm.STEPS_IN, pm.STEPS_OUT, data_file)
    # if pm.N_FEATURES > 1:
    xx = xx.reshape((xx.shape[0], xx.shape[1], pm.N_FEATURES))
    log.debug("Working with", xx.shape, " ", y.shape, "samples")

    return xx, y, scaler if keep_scaler else None

# noinspection PyTypeChecker
# scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
# scaler: MinMaxScaler = scaler.fit(float_inputs.reshape(-1, 1))
# dataset = trans_forward(scaler, float_inputs)
