# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:02:09 2017

标普500 加上中国隔夜股市的测试实验

@author: apple
"""
# %% load data
import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

import numpy as np

from TSLibrary import StockPredictionModel as spModel
from TSLibrary import StockPredictionSystem as spSystem
from TSLibrary import TimeSeriesAnalyzer as tsAnalyzer
from matplotlib import pyplot as plt

result_path = "result/LSTMRegressor_all_overnight"
isExists=os.path.exists(result_path)
if not isExists:
    os.makedirs(result_path)

# 标普500 数据较好
#data_path = '/Users/mayingrui/Desktop/暑研/代码！！！！！/实验代码/data/IndexData/^GSPC.csv'
data_path = '../data/CSI300_new.csv' 

FEATURES = ["Close", "Open", "High", "Low", "Volume", "Fluctuation"]
PRICE = "Close"
LABEL = "Target"
TIME = 'Date'
#load未处理的data,  数据,数量，dimension
index_set, index_num, features_size = spSystem.load_data(data_path)
#%% 循环处理
test_size = 0.3     # 测试集比例
list_data = spSystem.slide_windows(index_set, 2226 ,interval=(int)(2226 * test_size) )

# 每一个dataframe 作为一次独立实验
experiment = []
for df in list_data:

    #%% prepare data
    data = df.copy()
    data.index = range(0,len(data))
    origin_data = data['Close'].copy()
    # 生成目标数据
    data[LABEL] = spSystem.generate_target_regression(data['Close'])
    # 裁剪, 重新定义索引
    data = data[1:len(data)-1]
    data.index = range(0,len(data))

    #数据集划分
    train_data, val_data, test_data = spSystem.data_split(data, 0, test_size)
    train_data.index = range(0,len(train_data))
    val_data.index = range(0,len(val_data))
    test_data.index = range(0,len(test_data))
    
    #对所有的数据进行归一化
    scalers = dict()
    for feature in FEATURES:
        train_data[feature],scalers[feature] = spSystem.data_processor(train_data[feature])                
        test_data[feature] = scalers[feature].transform(np.array(test_data[feature]).reshape(-1,1)) 
    train_data[LABEL], scalers[LABEL] = spSystem.data_processor(train_data[LABEL])
    
    ###########################################################
    regressor = spModel.SPMDNNRegressor(FEATURES, LABEL)
    regressor.train_model(train_data, 5000)
    
    test_data = regressor.predict(test_data)
    res = test_data[['Date','Predict','Target']]

    # 缩放后的数据
    res['Predict_scale'] = res['Predict']
    # 数据逆缩放，得到原始数据集
    res['Predict'] = scalers['Target'].inverse_transform(np.array(res['Predict_scale']).reshape(-1,1))
    res['Target_scale'] = scalers['Target'].transform(np.array(res['Target']).reshape(-1,1))
        
    ###########################################################
    # 结果统计
    # 准确率
    da = spSystem.sDA(res['Predict'], res['Target'])
    # rmse需要使用标准化的数据
    rmse = spSystem.sRMSE(res['Predict_scale'], res['Target_scale'])
    # MAPE、CORR需要使用原始数据
    mape = spSystem.sMAPE(res['Predict'], res['Target'])
    # 相关系数需要对比差分值。否则没有意义
    corr = spSystem.sCorr(res['Predict'], res['Target'])
    stat = {"DA":da, "RMSE":rmse, "MAPE":mape, "CORR":corr}
           
    #图片绘制
    #%% 图形绘制
    fig, ax = plt.subplots(1, 1)
    fig_path = result_path + "/" + str(res['Date'][0])
    x = np.arange(len(res['Predict']))
    ax.plot(x, res['Predict'],'k', color='b')
    ax.plot(x, res['Target'],'k',color = 'g')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    fig.show()
    fig.savefig(fig_path)

     ###########################################################
    # 保存实验
    adict = dict()
    adict['result'] = res
    adict['scalers'] = scalers
    adict['train_data'] = train_data
    adict['val_data'] = val_data
    adict['test_data'] = test_data
    adict['statistic'] = stat
    experiment.append(adict)

#%% 数据存储
experiment_path = result_path + "/experiment_overnight.txt"
spSystem.save_experiment(experiment,experiment_path)