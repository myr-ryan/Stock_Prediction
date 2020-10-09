# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:30:44 2017
用于构建股指预测模型的接口
包括五个部分：
1. 数据处理
2. 模型搭建
3. 训练
4. 预测
5. 预警精度评估

@author: dell
"""


import pandas
from pandas import Series,DataFrame
import numpy as np
import pickle
import _pickle as cPickle
import pywt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd

#%%
###############################################################################
#   数据处理阶段
#   过程  数据加载 ->  (滑动窗口划分) -> (平稳化处理) -> 生成目标序列 ->
#         (特征选择方法) -> 去噪方法 -> 标准化方法 -> 数据集划分  
###############################################################################
def load_data(path):
    '''
    加载cvs数据
    :param path(string):  数据的路径。
    :return data(DataFrame):  股指数据集 (num, dim+1) 第一列为样本的时间戳。
            num(int32) : 样本数量
            dim(int32) : 特征维度
    '''
    data = pandas.read_csv(path, skipinitialspace=True)    
    num, dim = data.shape    
    dim = dim - 1
    return data,num,dim

def slide_windows(origin_data, k, interval = 1):    
    '''
    通过滑动窗口对数据集进行划分
    :param origin_data(DataFrame) : 原始数据集            
    :param k(int32)             :  滑动窗口大小
    :param interval(int32)      : 两个窗口的间隔,默认为1
    :return processed_data(list[dataframe, ..., dataframe]) : 划分扩增的数据     
    '''
    if not isinstance(origin_data, DataFrame):
        origin_data = DataFrame(origin_data)
    size = len(origin_data)
    origin_data.index = range(0,size)
    processed_data = []
    i = 0
    if k > size :
        k = size
        
    while i+k <= size :        
        processed_data.append(origin_data[i:i+k])
        i += interval + k        
    return processed_data
   
def difference_process(origin_data):
    '''
    序列差分处理，需要剔除首端数据  t_n - (t_n-1)  处理后剔除需要第一个值
    :param origin_data(Series)  :  原数据           
    :return diff_data(Series)   :  平稳化数据  
    '''
    if not isinstance(origin_data, Series):
        origin_data = Series(origin_data)
    origin_data.index = range(0,len(origin_data))
    diff = Series(len(origin_data), dtype='float64')
    for i in range(1, len(origin_data)):
        diff[i] = origin_data[i] - origin_data[i-1]
    return diff

# 待修正
def inverse_difference(diff_data, fundamental_value):
    '''
    逆差分处理，恢复原始数据
    :param  origin_data(Series)  :  差分后的数据
    :param  fundamental_value    :  初始值
    :return ori_data(Series)  :  原始数据
    '''
    if not isinstance(diff_data, Series):
        diff_data = Series(diff_data)
    diff_data.index = range(0,len(diff_data))
    ori_data = Series(len(diff_data), dtype='float64')
    ori_data[0] = diff_data[0] + fundamental_value
    for i in range(1, len(diff_data)):
        ori_data[i] = ori_data[i-1] + diff_data[i]
    ori_data = ori_data[1:]
    ori_data.index = range(len(ori_data))
    return ori_data

def difference_process_log(origin_data):
    '''
    序列差分处理，需要剔除首端数据  log(n/n-1) 可用于计算收益率
    :param origin_data(Series) : 原数据            
    :return diff_data(Series)  : 平稳化数据         
    '''
    if not isinstance(origin_data, Series):
        origin_data = Series(origin_data)        
    origin_data.index = range(0,len(origin_data))    
    diff = Series(len(origin_data), dtype='float64')
    for i in range(1, len(origin_data)):
        diff[i] = np.log(origin_data[i] / origin_data[i-1] )
    return diff

# 待修正
def inverse_difference_log(diff_data, fundamental_value):
    '''
    逆差分处理，恢复原始数据
    :param  origin_data(Series)  :  差分后的数据
    :param  fundamental_value    :  初始值
    :return ori_data(Series)  :  原始数据
    '''
    if not isinstance(diff_data, Series):
        diff_data = Series(diff_data)
    diff_data.index = range(0,len(diff_data))    
    ori_data = Series(len(diff_data), dtype='float64')
    ori_data[0] = np.power(np.e, diff_data[0])
    ori_data[0] = ori_data[0] * fundamental_value
    for i in range(1, len(diff_data)):
        ori_data[i] = np.power(np.e, diff_data[i])
        ori_data[i] = ori_data[i] * ori_data[i-1]
    return ori_data

def calculate_trend(origin_data, labels=[0,1], period=1):
    '''
    计算数据的趋势(两类): 
    if T >= T-1 :
        go up(+1)
    else
        go down(-1)
    :param origin_data(Series): 原始数据
    :parame labels  (array): 涨跌标签 默认[0,1]可选 [-1,1]
    :parame period  (int):  未来n日的收盘均值用于确定走势
    :return trend(Series) : 趋势数据
    '''
    if not isinstance(origin_data, Series):
        origin_data = Series(origin_data)
    trend = np.zeros(origin_data.size)
    for i in range(1, origin_data.size+1-period):
        data_mean = np.mean(origin_data[i:i+period])
        if data_mean >=  origin_data[i-1]:
            trend[i] = labels[1]
        elif data_mean < origin_data[i-1]:
            trend[i] = labels[0]
    return Series(trend)

def generate_target_regression(origin_data):
    '''
    生成目标序列,需要剔除末端数据     Y_t = X_(t+1)
    :param origin_data(Series)  : 原始数据        
    :return target(Series)      : 目标数据
    '''
    if not isinstance(origin_data, Series):
        origin_data = Series(origin_data)
    origin_data.index = range(0,len(origin_data))       
    target = Series(len(origin_data), dtype='float64')
    for i in range(0, len(origin_data)-1):
        target[i] = origin_data[i+1]
    return target

def denoise_method(origin_data, index_label ,method='wavelet'):
    '''
    使用去噪方法处理原始数据
    :param origin_data(DataFrame)  : 原始数据
    :param index_label(string) : DataFrame 中需要去噪的标签
    :param method(string)       : 减噪方法。 
                    *wavalet 小波去噪 默认
    :return origin_data(DataFrame) : 减噪后的数据
    '''

    def wavelet_denoise(data):
        haar = pywt.Wavelet('haar')
        # 分解
        cS2,cD2,cD1 = pywt.wavedec(data, haar, level = 2)
        # 重建
        rdata = pywt.waverec([cS2,None,None], haar)
        rdata = rdata[:len(rdata)-1]
        return rdata   
    
    DENOISE_METHOD_NAME = {
            "wavelet" : wavelet_denoise,
            }    
    
    if not isinstance(origin_data, DataFrame):
        origin_data = DataFrame(origin_data)
    if method not in DENOISE_METHOD_NAME:
        raise ValueError(
            "Denoise name should be one of [%s], you provided %s." %
            (", ".join(DENOISE_METHOD_NAME), method))
    denoise_data = DENOISE_METHOD_NAME[method](np.array(origin_data[index_label]))
    
    # 去掉末端数据
    origin_data = origin_data[:len(origin_data)-1]
    origin_data[index_label] = Series(denoise_data)
    
    return origin_data

def data_processor(origin_data, method='minmax'):
    '''
    原始数据的标准化过程
    :param origin_data(Series)  : 原始数据
    :param method(string)       : 标准化方法。 
                    *minmax 最小最大方法，数据缩放至0,1区间内。 默认
                    *standard z-score方法， 数据均值为0，标准差为1
    :return denoise_data(Series) : 标准化后的数据
    '''
    PROCESSOR_METHOD_NAME = {
            'minmax' ,
            'standard',
            }
    origin_data = np.array(origin_data).reshape(-1,1)
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0,1))
        return Series(scaler.fit_transform(origin_data)[:,0]), scaler
    elif method == 'standard':
        scaler = StandardScaler()
        return Series(scaler.fit_transform(origin_data)[:,0]), scaler    
    raise ValueError(
            "Processor name should be one of [%s], you provided %s." %
            (", ".join(PROCESSOR_METHOD_NAME), method))
     
def data_split(origin_data, val_size = 0, test_size = 0.3):
    '''
    将原始数据集划分为训练集、验证集、测试集
    :param origin_data (DataFrame): 原始数据集 2维
    :param val_size(float64)  : 验证集大小
    :param test_size(float64) : 测试集大小
    :return     train_data(DataFrame) : 训练集
                validation_data(DataFrame) : 验证集
                test_data(DateFrame)  ： 测试集
    '''
    if not isinstance(origin_data, DataFrame):
        origin_data = DataFrame(origin_data)
    origin_data.index = range(0,len(origin_data))   
    nval = int(round(len(origin_data) * (1 - val_size - test_size)))
    ntest = int(round(len(origin_data) * (1 - test_size)))
    train_data, validation_data, test_data = origin_data.iloc[:nval,:],origin_data.iloc[nval:ntest,:],origin_data.iloc[ntest:,:]
    train_data.index=range(len(train_data))
    validation_data.index = range(len(validation_data))
    test_data.index = range(len(test_data))
    return train_data, validation_data, test_data

def feature_select(origin_data):    
    print("Feature Select Method")

#%%  结果统计
###############################################################################
#   结果统计阶段
#       根均方差(RMSE), 评价绝对误差百分比(MAPE), 正确率(DA),
###############################################################################
def sRMSE(predicted, real):
    '''
    根均方差
    :param predicted(Series)  : 预测值
    :param real(Series)       : 真实值                    
    :return rmse(float64)     : RMSE
    '''
    a = np.array(predicted)
    b = np.array(real)
    rmse = np.sqrt(((a-b)**2).mean(axis = 0))
    return rmse

def sMAPE(predicted, real):
    '''
    平均误差百分比
    :param predicted(Series)  : 预测值
    :param real(Series)       : 真实值                    
    :return mape(float64)     : MAPE
    '''
    a = np.array(predicted)
    b = np.array(real)
    mape = ((b-a)/b).mean(axis = 0) 
    return abs(mape)

def sDA(predicted, real):
    '''
    正确率   1/N * ∑( (p_t - p_t-1) * (r_t - r_t-1) > 0 ? 1, 0)
    :param predicted(Series)  : 预测值
    :param real(Series)       : 真实值                    
    :return da(float64)       : DA
    '''
    a = np.array(predicted)
    b = np.array(real)
    trend_p = np.zeros(len(a))
    trend_r = np.zeros(len(b))
    for i in range(1, len(a)):
        if (a[i] - a[i-1]) > 0 :
            trend_p[i] = 1
        elif (a[i] - a[i-1]) < 0:
            trend_p[i] = 0
        elif (a[i] - a[i-1]) == 0:
            trend_p[i] = trend_p[i-1]
        if (b[i] - b[i-1]) > 0 :
            trend_r[i] = 1
        elif (b[i] - b[i-1]) < 0:
            trend_r[i] = 0
        elif (b[i] - b[i-1]) == 0:
            trend_r[i] = trend_r[i-1]    
            
            
    d = []
#    for i in range(1,len(a)):
#        if ( a[i] - a[i-1] ) * (b[i] - b[i-1]) > 0:
#            d.append(1)
#        elif ( a[i] - a[i-1] ) * (b[i] - b[i-1]) < 0 :
#            d.append(0)
#        else :
#            if(len(d) == 0):
#                d.append(0)
#            else:
#                d.append(d[len(d)-1])
    for i in range(len(trend_p)):
        if(trend_p[i] == trend_r[i]):
            d.append(1)
        else:
            d.append(0)
    d = np.array(d)
    da = d.sum() / d.size
    return da

def sCorr(predicted, real):
    '''
    预测值与真实值的相关系数 pearson方法  协方差/标准差之积
    :param predicted(Series)  : 预测值
    :param real(Series)       : 真实值                    
    :return correlation(float64)     : corr
    '''
    if not isinstance(predicted, Series):
        predicted = Series(predicted)
    if not isinstance(real, Series):
        real = Series(real)
    return predicted.corr(real)

def sClassificationResult(predicted, real):
    '''
    统计预测和真实值之间的二分类结果。 数据为1, 和非1
    :param predicted(Series): 预测值
    :param real(Series): 真实值
    :return dict{
                accuracy(float)
                
                fscore(float)
                
    '''
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(real)):
        if (predicted.iloc[i] == 1 and real.iloc[i] == 1):
            TP += 1
        elif (predicted.iloc[i] != 1 and real.iloc[i] != 1):
            TN += 1
        elif (predicted.iloc[i] == 1 and real.iloc[i] != 1):
            FP += 1
        elif (predicted.iloc[i] != 1 and real.iloc[i] == 1):
            FN += 1
        
    
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP +FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    res = {'accuracy': accuracy, 'precision': precision, 'recall':recall, 'fscore' : f_score}
    
    return res


    
# %% 实验结果输出
def save_experiment(obj, path):
    '''
    保存实验数据
    :param obj : 存储的对象
    :param path(string): 存储路径    
    '''
    with open(path,'wb') as f1:
        pickle.dump(obj, f1)
        f1.flush()
        f1.close()

def load_experiment(path):
    '''
    读取存储的实验
    :param path(string): 存储路径    
    :return obj : 存储的对象
    '''
    exper_data = pd.read_pickle(path)
    return exper_data
    
def merge_experiment(experiment):
    '''
    数据合并(仅是简单拼接， 需要扩展以特定键值为唯一索引进行合并)
    --------
    Experiment: 实验数据
    '''
    e_a = pd.DataFrame()
    for e in experiment:
        df = e['result']
        e_a = e_a.append(df)
    e_a.index = np.arange(len(e_a))
    return e_a
##
    

###############################################################################    
class StockPredictSystem:
   # def __init__(self):
        
    def data_preparation(self, path):
#         数据加载 -> (滑动窗口划分) -> (平稳化处理) -> 生成目标序列 -> 
#         (特征选择方法) -> 标准化方法 -> 去噪方法 -> 数据集划分  
        self.data_path = path
        self.origin_data, self.data_count, self.data_dim = load_data(self.data_path)
        list_data = slide_windows(self.origin_data, len(self.origin_data)-1, 1)
        
        return list_data
    
    @property
    def _origin_data(self):
        return self.origin_data
    
    @property
    def _data_count(self):
        return self.data_count
    
    @property
    def _data_dimension(self):
        return self.data_dim
   
if __name__  == "__main__":
    path = "../data/IndexData/S&P500.csv"
    #数据初始化
    ori_data, data_c, data_d = load_data(path)
    # 以40天（约两个月）为间隔 每次500日（约2年）作为一组实验数据
    #list_data = slide_windows(ori_data, 500, 40)
    list_data = slide_windows(ori_data, len(ori_data))
    # 每一个dataframe 作为一次独立实验
    experiment = []
    for df in list_data:
        data = df.copy()
        data.index = range(0,len(data))
        origin_data = data['Close Price'].copy()
        # 对收盘价进行差分处理
        #data['Close Price'] = difference_process(data['Close Price'])
        #data['Close Price'] = difference_process_log(data['Close Price'])
        # 生成目标数据
        data['Target'] = generate_target_regression(data['Close Price'])
        # 去头去尾嘎嘣脆, 重新定义索引
        data = data[1:len(data)-1]
        data.index = range(0,len(data))
        #这里进行特征选择处理
        
        #对所有的数据进行归一化
        scalers = dict()
        for feature in data.columns:
            if feature != 'Ntime':
                data[feature],scalers[feature] = data_processor(data[feature])                
        
        #目标数据进行小波去噪
        data['dTarget'] = denoise_method(data['Target'])    
        #数据集划分
        train_data, val_data, test_data = data_split(data, 0.2, 0.2)
        
        ###########################################################
        #  建立模型
        data['Predict'] = data['Close Price']
        res = data[['Ntime','Predict','Target']]
        # 缩放后的数据
        res[['Predict_scale','Target_scale']] = res[['Predict','Target']]
        # 数据逆缩放，得到原始数据集
        res['Predict'] = scalers['Target'].inverse_transform(res['Predict'])
        res['Target'] = scalers['Target'].inverse_transform(res['Target'])
         
        ###########################################################
        # 结果统计
        # 准确率
        da = sDA(res['Predict_scale'], res['Target_scale'])
        # rmse需要使用标准化的数据
        rmse = sRMSE(res['Predict_scale'], res['Target_scale'])
        # MAPE、CORR需要使用原始数据
        mape = sMAPE(res['Predict'], res['Target'])
        # 相关系数需要对比差分值。否则没有意义
        corr = sCorr(res['Predict'], res['Target'])        
        #图片绘制
        res[['Predict','Target']].head(100).plot()
        
        ###########################################################
        # 保存实验
        adict = dict()
        adict['data'] = data
        adict['scalers'] = scalers
        adict['train_data'] = train_data
        adict['val_data'] = val_data
        adict['test_data'] = test_data
        experiment.append(adict)
 