import json

import numpy as np
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    newdata = np.array([])
    newdata = newdata.reshape(0, 5)
    with open('data/data.json') as f:
        data = json.load(f)
        i = 0
        for item in data:
            start = int(item['start_time'])
            end = int(item['end_time'])
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
    for i in range(0, len(newdata), 10):
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

    wsize = 2
    # smoothing with moving average
    for _ in range(1):
        for i in range(0, len(newdata)):
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

    global_mid = int(newdata[0][2])
    # split into
    with open(f'data/{global_mid}_all.csv', 'w') as f:
        f.write("timestamp,empty1,empty2,machine_id,cpu,mem,empty3\n")
        i = 1
        for item in newdata:
            start, end, mid, cpu, mem = item
            f.write(f"{i},,,{mid},{cpu},{mem},{start},{end}\n")
            i += 1

    # split into pieces of three days
    for i in range(0, len(newdata), 3 * 288):
        with open(f'data/{global_mid}_{i//3}.csv', 'w') as f:
            f.write("timestamp,empty1,empty2,machine_id,cpu,mem,empty3\n")
            j = 1
            for item in newdata[i:i+3*288]:
                start, end, mid, cpu, mem = item
                f.write(f"{j},,,{mid},{cpu},{mem},{start},{end}\n")
                j += 1


    # plot a day at a time
    # for i in range(0, len(newdata), 12 * 24):
    #     dd = newdata[i:i+12*24]
    #     x = [item[0] for item in dd]
    #     y = [item[4] for item in dd]
    #     plt.figure(figsize=(30, 10))
    #     plt.plot(x, y)
    #     plt.savefig(f"data/plot{i}.png")
    #     plt.close()
