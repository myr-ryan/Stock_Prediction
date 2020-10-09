# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:50:47 2018
 在原有LSTM模型的基础上加上了BatchNormalization 方法
 参考 https://github.com/OlavHN/bnlstm/blob/master/lstm.py
 对应论文 Recurrent Batch Normalization
@author: dell
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn
#%% 普通的LSTM节点层
class LSTMCell(RNNCell):    #定义的LSTMCell 继承与Tensorflow 的 RNNCell 在https://blog.csdn.net/mydear_11000/article/details/52414342 有较为详尽的说明
    
    def __init__(self, num_units):
        self.num_units = num_units
        
    @property
    def state_size(self):
        return (self.num_units, self.num_units)
    
    @property
    def output_size(self):
        return self.num_units
    
    def __call__(self, x, state, scope=None):
        # cell 代表着一个隐层
        # cell 的用法 (output, next_state) = call(input, state)

        c, h = state  # 历史状态
        
        x_size = x.get_shape().as_list()[1]
        W_xh = tf.get_variable('W_xh', [x_size, 4 * self.num_units], initializer=orthogonal_initializer())              #TODO: 待理解
        W_hh = tf.get_variable('W_hh', [self.num_units, 4 * self.num_units], initializer=bn_lstm_identity_initializer(0.95))    #TODO: 待理解  
        bias = tf.get_variable('bias', [4 * self.num_units])
        
        # cell, input, forget, output 门的计算
        concat = tf.concat([x, h],1)
        W_both = tf.concat([W_xh, W_hh],0)
        hidden = tf.matmul(concat, W_both) + bias
        
        i,j,f,o = tf.split(hidden, 4, 1)
        new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        
        return new_h, (new_c, new_h)
        
        
#%% 加上BatchNormalization 的LSTM层
class BNLSTMCell(RNNCell):
    def __init__(self, num_units, training):
        self.num_units = num_units
        self.training = training

    @property
    def state_size(self):
        return (self.num_units, self.num_units)
    
    @property
    def output_size(self):
        return self.num_units
    
    def __call__(self, x, state, scope=None):
        # cell 的用法 (output, next_state) = call(input, state)
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)
######################### BatchNormalization #######################
            bn_xh = batch_norm(xh, 'xh', self.training)     # 对于输入数据进行Batch Normalization
            bn_hh = batch_norm(hh, 'hh', self.training)     # 对于历史数据进行Batch Normalization
####################################################################            
            hidden = bn_xh + bn_hh + bias
            i, j, f, o = tf.split(hidden, 4, 1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = batch_norm(new_c, 'c', self.training)

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)     
#%% initializer definition
def orthogonal(shape):
    '''
    生成正交的随机数矩阵
    对高斯分布的使用奇异值分解过程
    '''
    flat_shape = (shape[0], np.prod(shape[1:])) # np.prod 一维为原始逐个相乘
    a = np.random.normal(0.0, 1.0, flat_shape) # 正太分布初始化
    u, _, v = np.linalg.svd(a, full_matrices=False)  # linalg 线性函数库。 SVD 为奇异值分解 
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def orthogonal_initializer():
    '''
    定义了tensor中权值的一个正交方法初始过程， 返回函数对象。
    函数定义为：
    def _initializer(shape, dtype=tf.float32, partition_info=None):
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer

def bn_lstm_identity_initializer(scale):
    '''
    ??? 返回一个 拉伸4倍的矩阵， 第一部分为0矩阵， 第二部分为单位矩阵*scale 第三四部分为奇异值分解矩阵
    函数定义为：
    def _initializer(shape, dtype=tf.float32, partition_info=None):
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        size = shape[0]
        
        t = np.zeros(shape)
        t[:, size:size*2] = np.identity(size) * scale
        t[:, size*2:size*3] = orthogonal([size, size])
        t[:, size*3:] = orthogonal([size, size])
        return tf.constant(t, dtype)
    
    return _initializer

def batch_norm(x, name_scope, traning, epsilon = 1e-3, decay = 0.999):
    '''
    batch normalization 过程 
    TODO: 待理解
    '''
    with tf.variable_scope(name_scope): # 定义变量命名空间
        size = x.get_shape().as_list()[1]
        
        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size]) 
        
        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0]) # 权值的一阶矩和二阶中心距
        
        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))   # 引用赋值 pop_mean 为新值， train_mean_op 为旧值
        train_var_op  = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay ))
        
        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)
        
        return tf.cond(traning, batch_statistics, population_statistics) # 断言 如果traning 为真 执行batch_stat 否则 执行 population_stat

# 维度扩展 
def rnn_features_labels(features, labels, time_steps):
    """
    creates new data frame based on previous observation
      * example:
        f = [1, 2, 3, 4, 5]
        l = [2, 3, 4, 5, 6]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4], [4, 5]]
        -> labels == True [3, 4, 5, 6]
    """
    rnn_features = []
    rnn_labels = []
    for i in range(len(features) - time_steps + 1):
        try:
            rnn_labels.append(labels[i - 1 + time_steps].as_matrix())
        except AttributeError:
            rnn_labels.append(labels[i - 1 + time_steps])

        _features = features[i: i + time_steps]
        rnn_features.append(_features if len(_features.shape) > 1 else [[i] for i in _features])

    return np.array(rnn_features, dtype=np.float32), np.array(rnn_labels, dtype=np.float32)

if __name__ == '__main__':
    import uuid
    import os
    import time
    batch_size = 100
    hidden_size = 100
    output_class = 10
    train_epoches = 5000
    time_steps = 3
    features_size = 784
    dropout = 0.6
    
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    
#    x = tf.placeholder(tf.float32, [None, 784])
#    x_inp = tf.expand_dims(x, -1)
    x = tf.placeholder(tf.float32, [None, time_steps, features_size])
    training =tf.placeholder(tf.bool) # 用于在batch_norm 中选择使用statistics方法
    

#    lstm = LSTMCell(hidden_size)
    lstm = BNLSTMCell(hidden_size, training)
    lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, input_keep_prob=1.0, output_keep_prob=dropout)
    initialState = (
            tf.random_normal([batch_size-time_steps+1, hidden_size], stddev=0.1),
            tf.random_normal([batch_size-time_steps+1, hidden_size], stddev=0.1))
    
#    output, state = rnn.dynamic_rnn(lstm, x_inp, initial_state=initialState, dtype=tf.float32)
    output, state = rnn.dynamic_rnn(lstm, x, initial_state=initialState, dtype=tf.float32)
    _, final_hidden = state
    
    W = tf.get_variable('W', [hidden_size, output_class], initializer=orthogonal_initializer())
    b = tf.get_variable('b', [output_class])
    
    y = tf.nn.softmax(tf.matmul(final_hidden, W) + b)
    y_ = tf.placeholder(tf.float32,[None, output_class])
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
#    optimizer = tf.train.AdamOptimizer()
    '''
    gvs = optimizer.compute_gradients(cross_entropy)
    # capped_gvs?
    capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs] 
    train_step = optimizer.apply_gradients(capped_gvs)
    '''
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("xe_loss", cross_entropy)
    '''
    for(grad, var), (capped_grad, _) in zip(gvs, capped_gvs):
        if grad is not None:
            tf.summary.histogram('grad/{}'.format(var.name), capped_grad)
            tf.summary.histogram('capped_fraction/{}'.format(var.name),
                                 tf.nn.zero_fraction(grad-capped_grad))
            tf.summary.histogram('Weight/{}'.format(var.name), var)
     '''     
    merged = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        
        log_dir = "logs/" + str(uuid.uuid4())
        os.makedirs(log_dir)
        write = tf.summary.FileWriter(log_dir, sess.graph)
        current_time = time.time()
    
        for i in range(train_epoches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs,batch_ys = rnn_features_labels(batch_xs, batch_ys, time_steps)
            loss,_ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys, training:True})
            step_time = time.time() - current_time
            current_time = time.time()
            if i % 100 == 0:
                # 训练记录
                batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
                batch_xs, batch_ys = rnn_features_labels(batch_xs, batch_ys, time_steps)
                summary_str,acc = sess.run([merged,accuracy], feed_dict={x:batch_xs, y_:batch_ys, training: True} )
                write.add_summary(summary_str,i)
                print("=====step %d========: (loss: %f, acc: %f)"%(i, loss, acc))
        
