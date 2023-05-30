import logging as log
import os
import random

import numpy as np

from src.support import dt
from src import parameters as pm
from src.support.log import tqdm_wrapper


def fetch_datasets():
    gcd_folder = pm.GCD_FOLDER

    banlist = []
    if os.path.exists("banlist"):
        with open("banlist", "r") as f:
            for line in f:
                banlist.append(line.strip() + ".csv")
    else:
        with open("banlist", "w") as f:
            f.write("")

    if os.path.exists(f"{gcd_folder}/.DS_Store"):
        os.remove(f"{gcd_folder}/.DS_Store")

    available_folders = [i for i in os.listdir(gcd_folder) if "json" not in i and "npy" not in i and "cache" not in i]
    chosen_folder = random.choices(available_folders, k=1)[0]

    files = [i for i in os.listdir(os.path.join(f"{gcd_folder}", chosen_folder)) if "seq" in i]
    files = sorted([i for i in files], key=lambda x: int(x.split("_")[-1].split("-")[0]))

    total_size = pm.TEST_FILE_AMOUNT + pm.TRAIN_FILE_AMOUNT
    # select a random 4-file chunk that is adjacent
    starting_index = random.randint(0, len(files) - total_size)
    files = files[starting_index:starting_index + total_size]

    train_data = [os.path.join(chosen_folder, i) for i in files[:pm.TRAIN_FILE_AMOUNT]]
    test_data = [os.path.join(chosen_folder, i) for i in files[pm.TRAIN_FILE_AMOUNT:]]

    log.info("Verifying test files...")
    for file in test_data:
        if file in banlist:
            raise EnvironmentError(f"File {file} is banned. Please refrain from using it ever again.")
        if not os.path.exists(os.path.join(pm.GCD_FOLDER, file)):
            raise EnvironmentError(f"File {file} does not exist.")

    log.info("Verifying train files...")
    for file in train_data:
        if file in banlist:
            raise EnvironmentError(f"File {file} is banned. Please refrain from using it ever again.")
        if not os.path.exists(os.path.join(pm.GCD_FOLDER, file)):
            raise EnvironmentError(f"File {file} does not exist.")

    if len(train_data) == 0 or len(test_data) == 0:
        raise EnvironmentError(
            "Something went wrong. You need to specify at least one file for training and one for testing.")

    log.info("Training files: " + str(train_data))
    log.info("Testing files: " + str(test_data))

    return train_data, test_data


def load_data(file: str) -> tuple[np.ndarray, np.ndarray]:
    npycpu = os.path.join(pm.CACHE_FOLDER, file.split("/")[-1].replace(".csv", "_cpu.npy"))
    npymem = os.path.join(pm.CACHE_FOLDER, file.split("/")[-1].replace(".csv", "_mem.npy"))
    file = os.path.join(pm.GCD_FOLDER, file)

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


def get_power_from_sequence(param):
    # watts
    cpu_mappings = [32.0, 64.3, 76.9, 90.5, 107.0, 122.0, 140.0, 160.0, 186.0, 214.0, 235.0]
    mem_mappings = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 2.0, 4.0, 6.0, 8.0]

    cpu_usage = param[:, 0]
    mem_usage = param[:, 1]

    # transform average power over 5 minutes to actual consumption (Wh)
    cpu_power = np.interp(cpu_usage, np.arange(len(cpu_mappings)), cpu_mappings) * pm.GRANULARITY / 60
    mem_power = np.interp(mem_usage, np.arange(len(mem_mappings)), mem_mappings) * pm.GRANULARITY / 60

    return np.sum(cpu_power + mem_power) * pm.OVERHEAD


def split_sequence(sequence, past_len, future_len, steps_out, filename):
    # split the sequence into samples s.t.
    # X = [past_len, N_FEATURES]
    # y = [steps_out, 1]
    # where steps_out is calculated using get_power_from_sequence, and requires future_len samples
    # to be computed

    seq_len = len(sequence) - future_len - past_len + 1
    try:
        xx, y = np.ndarray(shape=(seq_len, past_len, pm.N_FEATURES)), np.ndarray(shape=(seq_len, steps_out))
    except ValueError as e:
        log.warning(f"Error while splitting sequence (might be too short?): {e}")
        return None, None

    simple_filename = filename.split("/")[-1].replace(".csv", "")
    desc = ",".join([str(a) for a in [past_len, future_len, steps_out, simple_filename, len(sequence)]])

    seq_filename_npy = pm.CACHE_FOLDER + '/' + "samples_" + desc + ".npy"
    seq_filename_y_npy = pm.CACHE_FOLDER + '/' + "samples_" + desc + "_y.npy"
    seq_filename = pm.GCD_FOLDER + '/' + "samples_" + desc

    if os.path.exists(seq_filename_npy) and os.path.exists(seq_filename_y_npy):
        log.info(f"Using cached sequences for {seq_filename}...")
        return np.load(seq_filename_npy), np.load(seq_filename_y_npy)
    elif os.path.exists(seq_filename_npy) or os.path.exists(seq_filename_y_npy):
        raise Exception(f"Only one of {seq_filename_npy} or {seq_filename_y_npy} exists!")
    else:
        log.info(f"Generating sequences for {seq_filename}...")
        for i in tqdm_wrapper(range(seq_len)):
            end_ix = i + past_len

            seq_x = sequence[i:end_ix]
            seq_y = get_power_from_sequence(sequence[end_ix:end_ix + future_len])

            xx[i] = seq_x
            y[i] = seq_y

        np.save(seq_filename_npy, xx)
        np.save(seq_filename_y_npy, y)

        return xx, y


def obtain_vectors(data_file: str | list[str]) -> (np.ndarray, np.ndarray):
    if isinstance(data_file, list):
        xx, y = np.ndarray(shape=(0, pm.STEPS_IN, pm.N_FEATURES)), np.ndarray(shape=(0, pm.STEPS_OUT))
        for file in data_file:
            xx_, y_ = obtain_vectors(file)
            if xx_ is None or y_ is None:
                continue
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

    future_len = pm.STEPS_IN // dt.WEEK_IN_DAYS
    xx, y = split_sequence(sequence=dataset, past_len=pm.STEPS_IN, future_len=future_len, steps_out=pm.STEPS_OUT,
                           filename=data_file)
    # log.debug("Working with", xx.shape, " ", y.shape, "samples")

    return xx, y
