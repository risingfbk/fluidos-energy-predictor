import json
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import src.secret_parameters as pms
from src.log import tqdm_wrapper

if __name__ == "__main__":
    name_start = pms.PARSED_FILE
    name_target = name_start.replace(".json", "").replace("_sorted", "")
    folder_start = "data/json"
    folder_target = f"data/{name_target}"

    os.makedirs(folder_target, exist_ok=True)

    print("Reading data from ", folder_start, name_start)
    newdata = np.array([])
    newdata = newdata.reshape(0, 5)
    with open(f'{folder_start}/{name_start}') as f:
        data = json.load(f)
        i = 0
        total_items = len(data)
        for item in tqdm_wrapper(data):
            start = int(item['start_time'])
            end = int(item['end_time'])

            if len(newdata) > 1 and (
                    start == end or start == newdata[-1][0] or end == newdata[-1][1]
                ):
                continue

            mid = int(item['machine_id'])
            cpu = float(item['random_sample_usage']['cpus'])
            mem = float(item['average_usage']['memory'])
            # make newdata an array of arrays
            newdata = np.append(newdata, np.array([[start, end, mid, cpu, mem]]), axis=0)

    # print(newdata)
    print("Length of newdata:", len(newdata))
    print("Each data point is 30 sec. Total time: ", len(newdata) * 30 / 60 / 60 / 24, "days")

    # Merge data points so that it becomes one every 5 minutes
    # each data point is the average of 10 data points
    newnewdata = np.array([])
    newnewdata = newnewdata.reshape(0, 5)
    for i in tqdm_wrapper(range(0, len(newdata), 10)):
        if i + 10 > len(newdata):
            break
        start, end, mid, cpu, mem = newdata[i]
        for j in range(1, 10):
            _, _, _, cpu_, mem_ = newdata[i + j]
            cpu += cpu_
            mem += mem_
        cpu /= 10
        mem /= 10
        newnewdata = np.append(newnewdata, np.array([[start, end, mid, cpu, mem]]), axis=0)
    newdata = newnewdata

    print("Length of newdata:", len(newdata))
    print("Each data point is 5 min. Total time: ", len(newdata) * 5 / 60 / 24, "days")

    print("Computing moving average...")

    wsize = 2
    # smoothing with moving average
    for _ in range(1):
        for i in tqdm_wrapper(range(0, len(newdata))):
            if i < wsize:
                newdata[i][3] = np.mean(newdata[:i + wsize, 3])
                newdata[i][4] = np.mean(newdata[:i + wsize, 4])
            elif i >= len(newdata) - wsize:
                newdata[i][3] = np.mean(newdata[i - wsize:, 3])
                newdata[i][4] = np.mean(newdata[i - wsize:, 4])
            else:
                newdata[i][3] = np.mean(newdata[i - wsize:i + wsize, 3])
                newdata[i][4] = np.mean(newdata[i - wsize:i + wsize, 4])

    # Scale cpu and mem between 0 and 1
    scaler = MinMaxScaler()
    newdata[:, 3] = scaler.fit_transform(newdata[:, 3].reshape(-1, 1)).reshape(-1)
    newdata[:, 4] = scaler.fit_transform(newdata[:, 4].reshape(-1, 1)).reshape(-1)

    print("Writing data to ", folder_target, name_target)

    # split into
    with open(f'{folder_target}/{name_target}_all.csv', 'w') as f:
        f.write("timestamp,empty1,empty2,machine_id,cpu,mem,empty3\n")
        i = 1
        for item in newdata:
            start, end, mid, cpu, mem = item
            f.write(f"{i},,,{mid},{cpu},{mem},{start},{end}\n")
            i += 1

    print("Splitting data into days...")

    # split into pieces of three days
    for i in tqdm_wrapper(range(0, len(newdata), 3 * 288)):
        day_start = i // 288
        day_end = (i + 3 * 288) // 288 - 1
        print(f"Splitting from t={day_start} to t={day_end}")
        with open(f'{folder_target}/{name_target}_days_{day_start}-to-{day_end}.csv', 'w') as f:
            f.write("timestamp,empty1,empty2,machine_id,cpu,mem,empty3\n")
            j = 1
            for item in newdata[i:i + 3 * 288]:
                start, end, mid, cpu, mem = item
                f.write(f"{j},,,{mid},{cpu},{mem},{start},{end}\n")
                j += 1

    os.makedirs(f"{folder_target}/plot", exist_ok=True)

    print("Plotting data...")
    # plot a day at a time
    for i in tqdm_wrapper(range(0, len(newdata), 12 * 24)):
        dd = newdata[i:i + 12 * 24]
        x = [item[0] for item in dd]
        y = [item[4] for item in dd]
        plt.figure(figsize=(30, 10))
        plt.plot(x, y)
        plt.savefig(f"{folder_target}/plot/{name_target}_plot_day_{i // (12 * 24)}.png")
        plt.close()
