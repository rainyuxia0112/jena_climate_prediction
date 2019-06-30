#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:51:33 2019

@author: rain
"""

# laoding data
import os
import pandas as pd
data = pd.read_csv('jena_climate_prediction/jena_climate_2009_2016.csv')
data.head()


# 查看这个dataframe的每一个col的missing data， 有多少个数据， 有多少个unique数据，以及其他的summary
# 在一开始要去确认 每一列的datatype, 建立两个list， 一个存连续性变量，一个存离散型变量
def summary(df):
    """
    return - 返回一个dataframe 其中包括：mean，std，分位数， missing data占比，数据总个数，每一列unique个数
    print - 是data info（包括数据类型）
    这个主要侧重的是连续型的变量
    """
    describe = df.describe()
    describe = describe.T
    total = len(df)
    describe['percentage of missing data'] = round((total - describe['count']) /total, 2)  #统计缺失值占比
    describe['count'] = total
    describe['number of unique'] = df.apply(lambda x: len(x.unique())) #应用在每一列上
    print (df.info())
    return (describe)

describe = summary(data)  #no missing data


# 画出温度随时间的变化
data.plot(x = 'Date Time' , y = 'T (degC)')

# 画出前10天的温度变化
data[ :1440].plot(x = 'Date Time' , y = 'T (degC)')


# 对数据进行标准化处理
## 对时间进行处理：datetime
data['Date Time'] = pd.to_datetime(data['Date Time'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.iloc[:200000, [1,2,3,4,5,6,7,8,9,10,11,12,13,14]])    # 针对col进行标准化处理

test_data = scaler.transform(data.iloc[200000:, [1,2, 3,4,5,6,7,8,9,10,11,12,13,14]])   # 对测试集进行相同变换

# 生成时间序列样本
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint( min_index + lookback, max_index, size=batch_size)
    else:
        if i + batch_size >= max_index:
        
            i = min_index + lookback
        rows = np.arange(i, min(i + batch_size, max_index)) 
        i += len(rows)
    samples = np.zeros((len(rows), lookback // step,
                        data.shape[-1])) 
    targets = np.zeros((len(rows),))
    for j, row in enumerate(rows):
        indices = range(rows[j] - lookback, rows[j], step) 
        samples[j] = data[indices]
        targets[j] = data[rows[j] + delay][1]
    yield samples, targets


lookback = 1440 
step = 6
delay = 144 
batch_size = 128
train_gen = generator(scaled_data, lookback=lookback, delay=delay, 
                      min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(test_data, lookback=lookback,
                    delay=delay, min_index=0, max_index=100000, step=step, batch_size=batch_size)
test_gen = generator(test_data, lookback=lookback,
                     delay=delay, min_index=100001, max_index=None, step=step, batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) //batch_size
test_steps = (len(test_data) - 100001 - lookback) //batch_size

# Build model
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, scaled_data.shape[-1])))  #就是14 （shape)
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae') 
history = model.fit_generator(train_gen,
steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)

