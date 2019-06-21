import pandas as pd
import numpy as np

import multiprocessing as mp
import time
import os
import csv

from scipy.interpolate import interp1d
from itertools import islice
from scipy import signal

from util.const import LINES_CHUNK, DATASET_PATH, g, COLS, NORM_COLS
import util.features as ft

#预处理（丢异常、滤波、插值）
def export_preprocess():

    for i in range(10):
        s = './' + DATASET_PATH + '/accData' + str(i) + '.txt'
        print(s)
        time = []
        ax = []
        ay = []
        az = []

        with open(s) as acc_file:
            acc_datas = acc_file.readlines()
            for k,acc_data in enumerate(acc_datas):
                t,x,y,z = [float(j) for j in acc_data.split(" ")]
                acc_datas[k] = [x,y,z]
                time.append(t)
                ax.append(x)
                ay.append(y)
                az.append(z)

        tmp=time[0]
        for j in range(len(time)):
            time[j]-=tmp

        time=np.array(time)
        ax=np.array(ax)
        ay=np.array(ay)
        az=np.array(az)

        #滤波
        b,a = signal.butter(4,0.4,'lowpass')
        ax = signal.filtfilt(b, a, ax)
        ay = signal.filtfilt(b, a, ay)
        az = signal.filtfilt(b, a, az)

        #三次样条插值
        fx=interp1d(time,ax,kind='cubic')
        fy=interp1d(time,ay,kind='cubic')
        fz=interp1d(time,az,kind='cubic')

        #丢弃前1000条包含异常值的数据
        new_time = [time[1000]]
        new_ax = [fx(new_time)]
        new_ay = [fy(new_time)]
        new_az = [fz(new_time)]

        preprocess=[]
        preprocess.append([new_ax[0][0],new_ay[0][0],new_az[0][0]])

        for j in range(1,28500):
            new_time.append(new_time[j-1]+20)
            new_ax.append(fx(new_time[j-1]+20))
            new_ay.append(fy(new_time[j-1]+20))
            new_az.append(fz(new_time[j-1]+20))
            preprocess.append([new_ax[j],new_ay[j],new_az[j]])

        preprocess=np.array(preprocess)
        new_time=np.array([new_time])
        new_ax=np.array(new_ax)
        new_ay=np.array(new_ay)
        new_az=np.array(new_az)

        df=pd.DataFrame(preprocess)

        sname='./data/preprocess'+str(i)+'.csv'
        df.to_csv(sname, encoding='utf-8', index=False)

def export_csv():
    df = pd.DataFrame(columns=COLS)

    for i in range(10):
        s = './' + DATASET_PATH + '/preprocess' + str(i) + '.csv'
        print(s)
        df = df.append(process_file(s,i), ignore_index=True)

    #归一化
    df[NORM_COLS] = df[NORM_COLS].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    df.to_csv('proc.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

#特征工程
def process_wrapper(lines, fname,i):
    sensorVals = []
    with open(fname, 'r') as f:
        for line in lines:
            vals = line.split(',')
            try:
                sensorVals.append((float(vals[0]), float(vals[1]), float(vals[2])))
            except:
                pass

    results = []

    #平均值
    means = ft._avg(sensorVals)
    results.extend([means[0], means[1], means[2]])

    #标准差
    results.extend(ft._stddev(sensorVals, means))

    #绝对离差
    results.extend(ft._absdev(sensorVals, means))

    #矢量合成
    results.append(ft._resultant(sensorVals))

    #过零率
    results.extend(ft._zerocross(sensorVals))

    #最小值
    results.extend(ft._minima(sensorVals))

    #最大值
    results.extend(ft._maxima(sensorVals))

    #类别标签
    results.append(i)

    #柱状图
    results.extend(ft._hist(sensorVals, -1.5 * g, 1.5 * g))

    return results

#分组
def chunkify(fname, lines=LINES_CHUNK):
    with open(fname, 'r') as f:
        #跳过第一行index
        next(f)

        for chunk in zip(*[f] * lines):
            yield chunk

#多线程
def process_file(file,i):
    df = pd.DataFrame(columns=COLS)

    pool = mp.Pool(processes=None)
    jobs = []

    #创建作业
    for chunk in chunkify(file, LINES_CHUNK):
        jobs.append(pool.apply_async(process_wrapper, args=(chunk,file,i)))

    #等待所有作业完成
    for job in range(len(jobs)):
        df.loc[df.shape[0]] = jobs[job].get()

    pool.close()

    return df