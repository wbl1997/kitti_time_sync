import datetime as dt
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import random

tunix0 = 0

def load_timestamps(data_path, data='oxts'):
    """Load timestamps from file."""
    timestamp_file = os.path.join(
        data_path, data, 'timestamps.txt')

    # Read and parse the timestamps
    timestamps = []
    with open(timestamp_file, 'r') as f:
        count=0
        for line in f.readlines():
            # NB: datetime only supports microseconds, but KITTI timestamps
            # give nanoseconds, so need to truncate last 4 characters to
            # get rid of \n (counts as 1) and extra 3 digits
            t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            t = dt.datetime.timestamp(t)
            if count==0:
               global tunix0
               tunix0 = int(t)
            # print(float("."+line.split(".")[-1]))
            t = int(t-tunix0) + float("."+line.split(".")[-1])
            # print("{0:.9f}".format(t))
            timestamps.append(t)
            count+=1

    # Subselect the chosen range of frames, if any
    return timestamps

def save_sync_timestamps(data_path, timestamps, data='oxts'):
    """Load timestamps from file."""
    timestamp_file = os.path.join(
        data_path, data, 'timestamps_sync.txt')

    # write timestamps
    with open(timestamp_file, 'w') as f:
        count=0
        for t in timestamps:
            # print(str("{0:.9f}".format(t)).split(".")[-1])
            t_ = dt.datetime.fromtimestamp(t+tunix0)
            dt_string = t_.strftime("%Y-%m-%d %H:%M:%S.%f")
            dt_string = dt_string[:-6]+str("{0:.9f}".format(t)).split(".")[1]
            # dt_string=dt_string[:-6]+temp01[count]
            count+=1
            f.write(dt_string+"\n")

def time_sync_check(x, timestamps):
    '''
    use line fitting to detect err time, $rate_inlier$ is strict. formula: t-t0 = fre_dt*(x-x0)
    x         : the index     
    timestamps: the time    
    fre_dt    : frequency of IMU, can be auto-estimated by code
    '''
    sigma_rate = 5   # sigma_inlier = sigma_rate*fre_dt

    last_timestamp = timestamps[:-1]
    curr_timestamp = timestamps[1:]
    dt = np.array(curr_timestamp - last_timestamp) #计算前后帧时间差
    fre_dt = np.median(dt)
    print("fre_dt (origin dt_median) is: {}".format(fre_dt))

    plt.figure(1)
    plt.plot(x[:-1], dt, 'r-', label='imu') # 可视化ＩＭＵ的时间戳

    list_temp = range(0, len(dt))
    sample_ind = random.sample(list_temp, 200)
    
    count=-1
    x_out = []
    x_in = []

    # check with line: t-t0 = fre_dt*(x-x0)
    sigma_inlier = sigma_rate*fre_dt
    for i in range(len(sample_ind)):
        t0 = timestamps[sample_ind[i]]
        x0 = x[sample_ind[i]]
        err = timestamps-t0-fre_dt*(x-x0)
        if x[abs(err)<sigma_inlier].shape[0]>count:
            count = x[abs(err)<=sigma_inlier].shape[0]
            x_out = x[abs(err)>sigma_inlier]
            x_in = x[abs(err)<=sigma_inlier]

    # refine the line formula and compute err
    t0 = timestamps[x_in].mean()
    x0 = x[x_in].mean()
    fre_dt = (t0-timestamps[x_in][0])/(x0-x[x_in][0])
    t_ = t0+fre_dt*(x-x0)
    print("refined fre_dt : {}, x0: {}".format(fre_dt, x0))

    # plt.figure(1000)
    # plt.plot(x, timestamps, 'r-', label='imu') # 可视化IMU的时间戳
    # plt.plot(x, t_, 'g-', label='imu') # 可视化IMU的时间戳

    err = timestamps-t0-fre_dt*(x-x0)
    err_th = sigma_inlier
    # err_th = rate_err * np.median(abs(err))
    # print("median: {},  mean: {}".format(np.median(abs(err)), np.mean(abs(err))))
    # print("err_th (rate_err*median) : {}".format(err_th))
    x_out = x[abs(err)>err_th]
    x_in = x[abs(err)<=err_th]
    print("the outlier index: ")
    print(x[abs(err)>err_th])

    plt.figure(2)
    plt.plot(x, abs(err), 'g-', label='imu') # 可视化IMU的时间戳
    plt.scatter(x_out, abs(err)[x_out], c='r')
    plt.title('index vs fit_err: err_th={}'.format(err_th))

    # modify
    for i in range(x_out.shape[0]):
        timestamps[x_out[i]] = fre_dt*(x[x_out[i]]-x0) + t0
    
    # recalculate dt and test
    last_timestamp = timestamps[:-1]
    curr_timestamp = timestamps[1:]
    dt = np.array(curr_timestamp - last_timestamp) #计算前后帧时间差
    plt.figure(1)
    plt.plot(x[:-1], dt, 'go', label='imu') # 可视化ＩＭＵ的时间戳
    plt.title('index vs dt (red-origin, green-modify)')

if __name__ == "__main__":
    data_path = "/media/wbl/KESU/data/kitti_sync/data/06/2011_09_30_drive_0020_extract"
    timestamps = np.array(load_timestamps(data_path), dtype="float64")
    x = np.arange(0, len(timestamps))
    time_sync_check(x, timestamps)
    save_sync_timestamps(data_path, timestamps)
    plt.show()

