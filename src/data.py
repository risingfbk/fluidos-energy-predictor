import logging as log
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src import parameters as pm
from src.log import tqdm_wrapper


def load_data(file: str) -> tuple[np.ndarray, np.ndarray]:
    npycpu = os.path.join(pm.CACHE_DIR, file.split("/")[-1].replace(".csv", "_cpu.npy"))
    npymem = os.path.join(pm.CACHE_DIR, file.split("/")[-1].replace(".csv", "_mem.npy"))
    file = os.path.join(pm.DATA_DIR, file)

    if os.path.exists(npycpu) and os.path.exists(npymem):
        log.info(f"Reading cached data from {npycpu} and {npymem}")
        return np.load(npycpu), np.load(npymem)
    elif os.path.exists(npycpu) or os.path.exists(npymem):
        raise Exception(f"Only one of {npycpu} and {npymem} exists")

    log.info(f"Reading data from {file}")

    ret_cpu = []
    ret_mem = []
    # ret_ts = []
    with open(file, 'r') as f:
        # data be like
        # timestamp,container_id,task_id,node_id,cpu,mem,idk
        # 38154,69b..,47f4...,K8ad..., 0.052437,0.3073234,300000

        for line in tqdm_wrapper(f):
            if 'timestamp' in line:
                continue

            fts, _, _, _, cpu, mem, *_ = line.split(',')

            # depending on the file, the timestamp can be in different positions
            # timestamp be like: filename:12341513
            if ":" in fts:
                _, timestamp = fts.split(':')
            else:
                timestamp = fts

            try:
                # ret_ts.append(int(timestamp))
                ret_cpu.append(float(cpu))
                ret_mem.append(float(mem))
            except ValueError as e:
                log.warning(f"Error while parsing {line}: {e}")
                pass
            # if len(res) > threshold + 1:
            #    break

    # res.sort(key=lambda x: x[0])
    # float_inputs = [x[1] for x in res]

    # log.debug("Computing moving average...")
    # mv_avg_window = pm.SMOOTH_WINDOW
    # mv_avg_runs = pm.SMOOTH_RUNS

    # for _ in range(mv_avg_runs):
    #     for i in tqdm_wrapper(range(len(float_inputs) - mv_avg_window)):
    #        new_inputs.append(np.mean(float_inputs[i:i + mv_avg_window]))

    # float_inputs = np.array(float_inputs)
    # np.save(npyfile, float_inputs)
    # np.save(file + "_ts.npy", np.array(ts))

    # ret_ts = np.array(ret_ts)
    ret_cpu = np.array(ret_cpu)
    ret_mem = np.array(ret_mem)

    np.save(npycpu, ret_cpu)
    np.save(npymem, ret_mem)

    return ret_cpu, ret_mem


# def trans_forward(scaler: MinMaxScaler, arr):
#     if len(arr.shape) == 1:
#         arr = arr.reshape(-1, 1)
#     return scaler.transform(arr)


# def trans_back(scaler: MinMaxScaler, arr):
#     if len(arr.shape) == 1:
#         arr = arr.reshape(-1, 1)
#     return scaler.inverse_transform(arr)


def split_sequence(sequence, n_steps_in, n_steps_out, filename):
    seq_len = len(sequence) - n_steps_out - n_steps_in + 1
    xx, y = np.ndarray(shape=(seq_len, n_steps_in, pm.N_FEATURES)), np.ndarray(shape=(seq_len, n_steps_out * pm.N_FEATURES))

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
        for i in tqdm_wrapper(range(seq_len)):
            end_ix = i + n_steps_in

            seq_x = sequence[i:end_ix]
            seq_y = sequence[end_ix:end_ix + n_steps_out]
            # reshape to a flat vector
            seq_y = seq_y.reshape(n_steps_out * pm.N_FEATURES)
            # [np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_TARGET),
            #  np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_LOWER),
            #  np.percentile(sequence[end_ix:end_ix + ywindow], pm.LSTM_UPPER)]
            xx[i] = seq_x
            y[i] = seq_y


        np.save(seq_filename_npy, xx)
        np.save(seq_filename_y_npy, y)

        return xx, y


def obtain_vectors(data_file: str | list[str]) -> (np.ndarray, np.ndarray):
    if isinstance(data_file, list):
        xx, y = np.ndarray(shape=(0, pm.STEPS_IN, pm.N_FEATURES)), np.ndarray(shape=(0, pm.STEPS_OUT * pm.N_FEATURES))
        for file in data_file:
            xx_, y_ = obtain_vectors(file)
            xx = np.concatenate((xx, xx_), axis=0)
            y = np.concatenate((y, y_), axis=0)
        return xx, y

    cpu, mem = load_data(data_file)
    dataset = []
    for series in range(len(cpu)):
        dataset.append([cpu[series], mem[series]])
    dataset = np.array(dataset)
    if len(dataset.shape) == 1:
        dataset = dataset.reshape(-1, 1)

    xx, y = split_sequence(dataset, pm.STEPS_IN, pm.STEPS_OUT, data_file)
    log.debug("Working with", xx.shape, " ", y.shape, "samples")

    return xx, y
