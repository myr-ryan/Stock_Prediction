# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:19:06 2017
stock predict model
股指的预测模型

@author: dell
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import learn as tflearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
#from statsmodels.tsa.arima_model import ARIMA

import numpy as np
import pandas as pd
from . import data_processing

from keras.models import Sequential
from keras.layers import Dense
from .dbn.tensorflow import SupervisedDBNRegression

from .lstm import lstm_model
from .bnLSTM import BNLSTMCell
class SPMDNNRegressor:
    '''
    借由tensorflow Estimator 实现的DNN回归方法
    使用的数据格式为DataFrame
    
    参数
    ------------
    FEATURES : 数据特征名
    LABEL : 数据目标标签名
    model_dir: 模型存储的位置，默认为None
    '''
    def __init__(self, FEATURES, LABEL, model_dir=None, hidden_units=[20,10], activation_fn = tf.nn.relu, optimizer='Adam', dropout=None):
        '''
        定义创建模型中的参数，及构建模型
        '''
        self.FEATURES = FEATURES
        self.LABEL = LABEL
        self.model_dif = model_dir
        # 模型定义        
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        self.optimizer=optimizer
        self.build_model()
    

    def get_input_fn(self, data_set, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.pandas_input_fn(
            x= pd.DataFrame({k: data_set[k].values for k in self.FEATURES}),
            y = pd.Series(data_set[self.LABEL].values),
            num_epochs=num_epochs,
            shuffle=shuffle)
    def build_model(self):
        #建立预测模型
        feature_cols = [tf.feature_column.numeric_column(k) for k in self.FEATURES]        
        self.model = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=self.hidden_units, activation_fn=self.activation_fn, 
                                               optimizer=self.optimizer, model_dir=self.model_dif, dropout = None)
        
    def train_model(self, training_set, steps):
        '''
        模型训练
        :param training_set (DataFrame) : 训练集
        :param step : 训练步数
        '''
        self.model.train(input_fn=self.get_input_fn(training_set), steps=steps)
        
    def predict(self, pred_set, predict_label='Predict'):
        '''
        预测
        :param pred_set(DataFrame) : 预测数据
        :param predict_label(string) : 预测数据的标签
        :return pred_set (DataFrame) : 预测值, 包括预测的序列
        '''
        y = self.model.predict(input_fn=self.get_input_fn(pred_set, num_epochs=1, shuffle=False))
        
        pred=np.array(list(p['predictions'] for p in y))
        pred = pd.Series(pred.reshape(-1))
        pred_set[predict_label] = pred
        pred_set = pred_set[:len(pred)]
        return pred_set

 
    
class SPMMLPRegressor:
    '''
    使用Keras 搭建MLP模型
    
    参数
    ------------
    FEATURES : 数据特征名
    LABEL : 数据目标标签名
    '''
    def __init__(self, FEATURES, LABEL):
        self.FEATURES = FEATURES
        self.LABEL = LABEL
        
        self.build_model()
        self.batch_size = 16
    def build_model(self):
        
        #建立预测模型
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=len(self.FEATURES), activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')
        
    def train_model(self, training_set, steps):
        '''
        模型训练
        :param training_set (DataFrame) : 训练集
        :param step : 训练步数
        '''
        # 数据类型切换
        trainX = []
        for i in self.FEATURES:
            trainX.append(training_set[i])
        trainY = training_set[self.LABEL]
        trainX = np.array(trainX).T
        trainY = np.array(trainY).T
        
        self.model.fit(trainX, trainY, nb_epoch=steps, batch_size=self.batch_size, verbose=2)
        
    def predict(self, pred_set, predict_label='Predict'):
        '''
        预测
        :param pred_set(DataFrame) : 预测数据
        :param predict_label(string) : 预测数据的标签
        :return pred_set (DataFrame) : 预测值, 包括预测的序列
        '''
        
        testX = []
        for i in self.FEATURES:
            testX.append(pred_set[i])
        testX = np.array(testX).T
        y = self.model.predict(testX)             
        
        y = pd.Series(y.reshape(-1))
        pred_set[predict_label] = y
        pred_set = pred_set[:len(y)]
        return pred_set
    
class SPMDBNRegressor:
    '''
    使用tensorflow 搭建的深度置信网络DBN
    
    参数
    ------------
    FEATURES : 数据特征名
    LABEL : 数据目标标签名
    '''
    def __init__(self, FEATURES, LABEL):
        self.FEATURES = FEATURES
        self.LABEL = LABEL
        
        self.hidden_units = [10,5]
        self.lr_rbm = 0.47
        self.lr = 0.001
        self.batch_size = 16
        self.activation_fn = 'sigmoid'
        self.rbm_epochs = 20
        self.steps=500
        self.build_model()
        
    def build_model(self):
        #建立预测模型
        self.model = SupervisedDBNRegression(hidden_layers_structure=self.hidden_units,
                                    learning_rate_rbm=self.lr_rbm,
                                    learning_rate=self.lr,
                                    n_epochs_rbm=self.rbm_epochs,
                                    n_iter_backprop=self.steps,
                                    batch_size=self.batch_size,
                                    activation_function=self.activation_fn)
        
    def train_model(self, training_set, steps):
        '''
        模型训练
        :param training_set (DataFrame) : 训练集
        :param step : 训练步数
        '''
        # 数据类型切换
        trainX = []
        for i in self.FEATURES:
            trainX.append(training_set[i])
        trainY = training_set[self.LABEL]
        trainX = np.array(trainX).T
        trainY = np.array(trainY).T
        
        self.model.fit(trainX, trainY)

        
    def predict(self, pred_set):
        '''
        预测
        :param pred_set(DataFrame) : 预测数据
        :return predict_value (Series) : 预测值
        
        '''
        testX = []
        for i in self.FEATURES:
            testX.append(pred_set[i])
        testX = np.array(testX).T
        y = self.model.predict(testX)      
        return pd.Series(y.reshape(-1))


# %%LSTM
class SPMLSTMRegressor:
    '''
    使用tensorflow中的搭建的深度置信网络DBN
    
    参数
    ------------
    FEATURES : 数据特征名
    LABEL : 数据目标标签名
    '''
    def __init__(self, FEATURES, LABEL, timesteps=3, rnn_layer=[{'num_units': 5}], batch_size = 64):
        self.FEATURES = FEATURES
        self.LABEL = LABEL
        
        self.timesteps = 3                # 预测的时间步长
        self.rnn_layer = [{'num_units': 5}] 
        self.dense_layer = None
        self.model_dir = None
        self.batch_size = 64
        
        self.build_model()
        
    def build_model(self):
        #建立预测模型
        self.model = tflearn.SKCompat(tflearn.Estimator(
                    model_fn=lstm_model(
                            self.timesteps,
                            self.rnn_layer,
                            self.dense_layer
                            ),
                        model_dir=self.model_dir
                    ))                 
                    
        
    def train_model(self, training_set, steps):
        '''
        模型训练
        :param training_set (DataFrame) : 训练集
        :param step : 训练步数
        '''
        # 数据类型切换
        trainX = training_set[self.FEATURES]
        trainY = training_set[self.LABEL]
        trainX,trainY = data_processing.rnn_features_labels(trainX, trainY, self.timesteps)
        #使用验证集来实现early stopping。
#        validation_monitor = learn.monitors.ValidationMonitor(X1['val'], Y1['val'],
#                                                      every_n_steps=PRINT_STEPS,
#                                                     early_stopping_rounds=1000)        
        self.model.fit(trainX, trainY, 
#              monitors=[validation_monitor], 
              batch_size=self.batch_size,
              steps=steps)      

        
    def predict(self, pred_set, pred_label = 'Predict'):
        '''
        预测
        :param pred_set(DataFrame) : 预测数据
        :param predict_label(string) : 预测数据的标签
        :return pred_set (DataFrame) : 预测值, 包括预测的序列
        
        '''
        pred_set.index = range(0,len(pred_set))
        testX = pred_set[self.FEATURES]
        testX = data_processing.rnn_data(testX, self.timesteps, include_end=True)
        
        #print(testX)
        y = self.model.predict(testX)
        
        # 索引调整， 忽略前 timestep步的数据
        y = pd.Series(y.reshape(-1))
        y.index=range(self.timesteps-1, len(pred_set))
        pred_set[pred_label] = y
        pred_set = pred_set[self.timesteps-1:]
        pred_set.index = range(0,len(pred_set))
        return pred_set

class SPMLSTMRegressorB:
    '''
    LSTM网络模型的另一种实现
    单前为单层结果
    模型无法复用
    参数
    ------------
    FEATURES : 数据特征名
    LABEL : 数据目标标签名
    lr : 学习率
    timesteps : RNN模型时间步长
    hidden_size : 模型的隐层节点个数
    keep_prob : dropout 节点保留率。
    layer_num : LSTM 层数  
    activation : LSTM 节点中的激活函数
    regression_activation : 顶层回归层的激活函数 默认为Relu
    norm_method: 节点的训练过程的Norm方法: {"no": 没有Norm方法 仅使用基准的LSTM节点(默认), "bn"：batch-norm方法, "ln"：layer-norm方法}
    '''
    def __init__(self, FEATURES, LABEL, input_size = 32, lr = 0.001, time_steps=3, hidden_size = 64, keep_prob = 1.0, layers_num = 1, 
                 activation=tf.nn.tanh, regression_activation=tf.nn.relu, norm_method="no"):
        # 初始化过程
        self.FEATURES = FEATURES
        self.LABEL= LABEL
        
        self.lr = lr
        self.input_size = input_size
        self.time_steps = time_steps
        self.hidden_size = 64
        self.class_num = 1 # class_num : 目标分类数。 当多分类或是输出是复数个重构因子的时候
        self.keep_prob = keep_prob
        self.layers_num = layers_num
        self.activation = activation
        self.r_activation = regression_activation
        self.norm = norm_method
        tf.reset_default_graph()
        self.build_model()
        
    def single_lstm(self, layer_size):
        # 单层LSTM节点的定义 这里是一次LSTM加上一次dropout warpper
        lstm_cell = rnn.BasicLSTMCell(num_units=layer_size, forget_bias=1.0, state_is_tuple=True, activation=self.activation)   # 使用的基础LSTM模型 后面紧接一次 Dropout层
        # TODO: 将Dropoout应用在RNN上有三个部分可以进行修改， 当前仅对输出做了控制
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.dropout_rate)
        return lstm_cell
    def single_bnlstm(self, layer_size, trainging=True):
        # 单层BNLSTM 节点并加上了Dropout Wrapper 
        bnlstm_cell = BNLSTMCell(layer_size, tf.constant(trainging))
        # TODO: 将Dropoout应用在RNN上有三个部分可以进行修改， 当前仅对输出做了控制
        bnlstm_cell = rnn.DropoutWrapper(cell=bnlstm_cell, input_keep_prob=1.0, output_keep_prob=self.dropout_rate)
        return bnlstm_cell
    def single_lnlstm(self, layer_size):
        lnlstm_cell = rnn.LayerNormBasicLSTMCell(layer_size, dropout_keep_prob=self.dropout_rate)        
        return lnlstm_cell
    
    def build_model(self):
        #建立模型
        self._X = tf.placeholder(tf.float32, [None, self.time_steps, self.input_size])
        
        self.Y = tf.placeholder(tf.float32, [None, self.class_num])
        self.dropout_rate = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        if self.norm == "no":
            mlstm_cell = rnn.MultiRNNCell([self.single_lstm(self.hidden_size) for _ in range(self.layers_num)])
#            mlstm_cell = rnn.MultiRNNCell([self.single_lnlstm(self.hidden_size) for _ in range(self.layers_num)])
            init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        elif self.norm == "ln":
            mlstm_cell = rnn.MultiRNNCell([self.single_lnlstm(self.hidden_size) for _ in range(self.layers_num)])
            init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        elif self.norm == "bn":
            # TODO : 加入多层会报错 TypeError: 'Tensor' object is not iterable.
#            mlstm_cell = rnn.MultiRNNCell([self.single_bnlstm(self.hidden_size) for _ in range(self.layers_num)])
            mlstm_cell = self.single_bnlstm(self.hidden_size)
            init_state = (
                    tf.random_normal([self.batch_size, self.hidden_size], stddev=0.1),
                    tf.random_normal([self.batch_size, self.hidden_size], stddev=0.1))
#            init_state = tf.contrib.rnn.LSTMStateTuple(tf.random_normal([self.batch_size, self.hidden_size]),tf.random_normal([self.batch_size, self.hidden_size]))
#            init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=self._X, initial_state=init_state, time_major=False)
        h_state = outputs[:, -1, :] 
        
        #顶层 逻辑回归
        W = tf.Variable(tf.truncated_normal([self.hidden_size, self.class_num], stddev=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.constant(0.1,shape=[self.class_num]), dtype=tf.float32)
        
        #   交叉熵， 用于分类
#        y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
#        cross_entropy = -tf.reduce_mean(self.Y * tf.log(y_pre))
#        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)
#        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(self.Y,1))
#        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#        # 回归器
        self.y_pre = self.r_activation(tf.matmul(h_state, W) + bias)
        # 目标函数魔改版 对多维向量取均值
#        self.loss = tf.reduce_mean(np.mean(tf.pow(self.Y - self.y_pre, 2)))
        self.loss = tf.reduce_mean(tf.pow(self.Y - self.y_pre, 2))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session() # TODO : 图需要初始化 
    
    def train_model(self, training_set, steps , show_steps = 50): 
        # 训练模型
        '''
        TODO : 训练集 shape 验证  Batch 训练
        raise ValueError('training_set shape can't match model input size')
        '''
        # 训练集划分
        trainX = training_set[self.FEATURES]
        trainY = training_set[self.LABEL]
        trainX,trainY = data_processing.rnn_features_labels(trainX, trainY, self.time_steps)
        trainY = trainY.reshape((-1,1))
#        
        sess = self.sess
        sess.run(tf.global_variables_initializer())   
        for i in range(steps):

            if (i+1) % show_steps == 0:
                train_accuracy = sess.run(self.loss, feed_dict={
                        self._X:trainX,self.Y: trainY, self.dropout_rate: self.keep_prob, self.batch_size: len(trainX)})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
                print(" step %d, training loss %g" % ((i+1), train_accuracy))
            sess.run(self.train_op, feed_dict={self._X: trainX, self.Y: trainY, self.dropout_rate: self.keep_prob, self.batch_size: len(trainX)})

    def predict(self, pred_set, pred_label = 'Predict'):
        # 预测
        pred_set.index = range(0,len(pred_set))
        testX = pred_set[self.FEATURES]
        testX = data_processing.rnn_data(testX, self.time_steps, include_end=True)
        
        y = self.sess.run(self.y_pre, feed_dict={self._X: testX, self.dropout_rate: 1.0, self.batch_size:len(testX)})

#        #应对多重构参数作为输出的情况
#        if self.class_num > 1:
#            return pred_set, y
        y = pd.Series(y.reshape(-1))
        y.index=range(self.time_steps-1, len(pred_set))
        pred_set[pred_label] = y
        pred_set = pred_set[self.time_steps-1:]
        pred_set.index = range(0,len(pred_set))
        return pred_set
#%% 分类模型
class SPMDNNClassifier:
    '''
    借由tensorflow Estimator 实现的DNN分类
    使用的数据格式为DataFrame
    
    参数
    ------------
    FEATURES : 数据特征名
    LABEL : 数据目标标签名
    model_dir: 模型存储的位置，默认为None
    '''
    def __init__(self, FEATURES, LABEL, model_dir=None, hidden_units=[30,20,20]):
        '''
        定义创建模型中的参数，及构建模型
        '''
        self.FEATURES = FEATURES
        self.LABEL = LABEL
        self.model_dif = model_dir
        # 模型定义        
        self.hidden_units = hidden_units
        self.activation_fn = tf.nn.relu
        self.lr = 0.1
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.build_model()
    

    def get_input_fn(self, data_set, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.pandas_input_fn(
            x= pd.DataFrame({k: data_set[k].values for k in self.FEATURES}),
            y = pd.Series(data_set[self.LABEL].values),
            num_epochs=num_epochs,
            shuffle=shuffle)
        
    def build_model(self):
        #建立预测模型
        feature_cols = [tf.feature_column.numeric_column(k) for k in self.FEATURES]        
        self.model = tf.estimator.DNNClassifier(feature_columns=feature_cols, hidden_units=self.hidden_units, activation_fn=self.activation_fn,
                                                n_classes = 2, optimizer=self.optimizer, model_dir=self.model_dif)
        
    def train_model(self, training_set, steps):
        '''
        模型训练
        :param training_set (DataFrame) : 训练集
        :param step : 训练步数
        '''
        self.model.train(input_fn=self.get_input_fn(training_set), steps=steps)
        
    def predict(self, pred_set, predict_label='Predict'):
        '''
        预测
        :param pred_set(DataFrame) : 预测数据
        :param predict_label(string) : 预测数据的标签
        :return pred_set (DataFrame) : 预测值, 包括预测的序列
        '''
        y = self.model.predict(input_fn=self.get_input_fn(pred_set, num_epochs=1, shuffle=False))
        pred=np.array(list(p['classes'] for p in y), dtype=float)
        pred = pd.Series(pred.reshape(-1))
        pred_set[predict_label] = pred
        pred_set = pred_set[:len(pred)]
        return pred_set
#%% 传统机器学习方法进行建立预测模型
        
class SPMRandomForestClassifier:
    '''
    借由sklearn 中的RandomFrost进行分类
    使用的数据格式为DataFrame
    
    参数
    ------------
    FEATURES : 数据特征名
    LABEL : 数据目标标签名
    estimator: 随机森林所使用的大小
    '''
    def __init__(self, FEATURES, LABEL,estimator=20):
        '''
        定义创建模型中的参数，及构建模型
        '''
        self.FEATURES = FEATURES
        self.LABEL = LABEL
        # 模型定义        
        self.estimator = estimator
        self.model = RandomForestClassifier(n_estimators=estimator)
        
    def train_model(self, training_set, steps=-1):
        '''
        模型训练
        :param training_set (DataFrame) : 训练集
        :param steps(int): 无效参数
        '''
        feature = training_set[self.FEATURES]
        label = training_set[self.LABEL]
        self.model.fit(feature, label)
    
    def predict(self, pred_set, predict_label='Predict'):
        '''
        预测
        :param pred_set(DataFrame) : 预测数据
        :param predict_label(string) : 预测数据的标签
        :return pred_set (DataFrame) : 预测值, 包括预测的序列
        '''
        pred_set[predict_label] = self.model.predict(pred_set[self.FEATURES])
        return pred_set

class SPMSVRRegression:
    '''
    使用SVR模型建立回归模型
    使用的数据格式为DataFrame
    
    参数
    ------------
    FEATURES : 数据特征名
    LABEL : 数据目标标签名
    C(float): Penalty parameter C of the error term.
    gamma(float): Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
    '''
    def __init__(self, FEATURES, LABEL, C=100, gamma=10.0):
        self.FEATURES = FEATURES
        self.LABEL = LABEL
        self.model = SVR(C=C,gamma=gamma)
        
    def train_model(self, training_set,steps=-1):
        '''
        模型训练
        :param training_set (DataFrame) : 训练集
        :param steps(int): 无效参数
        '''
        features = training_set[self.FEATURES]
        label = training_set[self.LABEL]
        self.model.fit(features, label)
        
    def predict(self, pred_set, predict_label='Predict'):
        '''
        预测
        :param pred_set(DataFrame) : 预测数据
        :param predict_label(string) : 预测数据的标签
        :return pred_set (DataFrame) : 预测值, 包括预测的序列
        '''
        pred_set[predict_label] = self.model.predict(pred_set[self.FEATURES])
        return pred_set

class SPMLinearPredictor:
    '''
    使用线性分析模型 ARIMA 进行 预测
    参数
    参考：https://machinelearningmastery.com/make-manual-predictions-arima-models-python/
    ------------
    FEATURES : 数据特征名 在ARIMA 模型中只有一维数据 模型只对自身序列进行建模
    LABEL : 数据目标标签名 不会使用到目标标签
    p (float) : 自回归项
    d (float) : 差分项
    q (float) : 移动均值项
    '''
    def __init__(self, FEATURES, LABEL, p, d, q):
        self.series_name = FEATURES[0]
        self.p = p
        self.q = q
        self.d = d
  
    def difference(self, dataset):
    	diff = list()
    	for i in range(1, len(dataset)):
    		value = dataset[i] - dataset[i - 1]
    		diff.append(value)
    	return np.array(diff)
    def predictH(self, coef, history):
    	yhat = 0.0
    	for i in range(1, len(coef)+1):
    		yhat += coef[i-1] * history[-i]
    	return yhat    
    
    def train_model(self, training_set,steps = -1):
        '''
        ARIMA 模型不需要进模型训练
        :param training_set (DataFrame) : 训练集
        :param steps(int): 无效参数
        '''
        self.price_series = [x for x in training_set[self.series_name]]

        
        
    def predict(self, pred_set, predict_label='Predict'):
        '''
        预测
        :param pred_set(DataFrame) : 预测数据
        :param predict_label(string) : 预测数据的标签
        :return pred_set (DataFrame) : 预测值, 包括预测的序列
        '''
        # 待删
        rand_arr = np.random.randn(len(pred_set))
        predictions = list()
        for t in range(pred_set.shape[0]):
            self.price_series.append(pred_set[self.series_name][t])
            model = ARIMA(self.price_series, order=(self.p, self.d, self.q))
            
            # 方法1
            model_fit = model.fit(trend='nc', disp=False)
            ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
            resid = model_fit.resid
            diff = self.difference(self.price_series)
            y_hat = self.price_series[-1] + self.predictH(ar_coef, diff) + self.predictH(ma_coef, resid)
            y_hat = y_hat + y_hat * 0.02 * rand_arr[t]
            predictions.append(y_hat)
            
            
##            #方法2
#            model_fit = model.fit(disp=0)
#            output = model_fit.forecast()
#            y_hat = output[0]
#            predictions.append(y_hat)
        
        pred_set[predict_label] = pd.Series(predictions)
        return pred_set
        
#class SPMRandomWalkPredict:
#    '''
#    随机游走模型用于预测过程
#    TODO: 通过价格数据上添加一个高斯分布的随机值
#    
#    参数：
#        scale ： (float32) 随机分布的跨度值        
#    '''
#    def __init__(self, scale):
#        