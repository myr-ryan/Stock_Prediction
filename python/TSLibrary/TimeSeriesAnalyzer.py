# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:18:57 2018
Time series analysis

时序分析相关函数
根据互信息确定时延τ（绘图判断MI是否能取得极小值）
根据Cao虚假近邻算法确定扩展维度m(返回一维数组E1，当d>d0是 E1不在变化。d0+1为最小嵌入维度)
@author: dell
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score 
# %% 相空间重构过程

#################### 一维数据求解时延与扩展维度 #################################


        ######################计算时延#############################
def calculate_mutual_info(targ, pred, spacing = 200):
    '''
    使用等间距法计算S,Q 的概率从而计算互信息
    ----------------------------------------
    :param targ(array) : 源序列
    :param pred(array) : 生成序列
    :param spacing(int,default=200) : 划分区间数量，默认200 
    '''

    s = np.array(targ)
    q = np.array(pred)
    if len(s) != len(q):
        raise ValueError("源序列与生成序列长度不一致")
    # 计算边界点
    xmin = np.min(s)
    xmax = np.max(s)
    ymin = np.min(q)
    ymax = np.max(q)
    # xy轴的边长
    width = xmax - xmin
    high = ymax - ymin
    # 当个网格的长宽
    gridw = width / spacing
    gridh = high / spacing
    # 计算网格内的落点
    Ps = np.zeros(spacing+1)
    Pq = np.zeros(spacing+1)
    Psq= np.zeros((spacing+1,spacing+1))
    
    for i in range(len(s)):
        tmpx = int((s[i]-xmin) / gridw) 
        tmpy = int((q[i]-ymin) / gridh)
        Ps[tmpx] += 1
        Pq[tmpy] += 1
        Psq[tmpx][tmpy] +=1
    # 源序列和生成序列的信息熵
    Hs = shan_entropy(Ps)
    Hq = shan_entropy(Pq)
    Hsq = shan_entropy(Psq)
    # 对应的互信息
    MI = Hs + Hq - Hsq
    MI = MI/np.sqrt(Hs * Hq)
    return MI

def calculate_mutual_info_2(s, q):
    # 互信息计算第二种方法
    return mutual_info_score(s,q)
def shan_entropy(c):
    # 计算香农熵
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H   
    
    
def calculate_delay(data, max_tao):
    '''
    互信息法求解时序时延τ。 将会绘制一个互信息变化图，根据图片选择最为合适的时延τ
    -------------------------------
    :param data (series): 时序目标序列的互信息量
    '''

    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    milist = []  # 用于绘图
    for i in range(1,max_tao+1):
        a = data[:len(data)-i]
        b = data[i:]
        milist.append(calculate_mutual_info(a, b))
    
    # 绘图观察 极小时延
#    plt.figure(figsize=(8,4),dpi=128)
#    plt.plot(np.arange(1,max_tao+1),milist)
#    plt.title("delay - mutual information")
    
    return milist

def define_delay(mi_list, max_tao):
    '''
    根据计算得到的时延数组，选择第一个极小值作为序列的时延间隔，若在max_tao前无
    极小值，则返回max_tao
    ---------------------------------------------------
    : param dlist (list) : 互信息数组，下标对应时延减一。
    : param max_tao  (int)  : 确定范围。
    : return tao : 确定的时延。
    TODO: 极小值的确定，仅使用第一个极小值较为模糊，具有一定的局限性，
    可能对于某些因子无法适用。
    '''
    if(max_tao >= len(mi_list)-1):
        max_tao = mi_list-1;
    for i in range(max_tao):
        if(mi_list[i] < mi_list[i+1]):
            return i+1    # 下标是从0开始所以要加1
    return max_tao
    ##############################计算扩展维度#########################
def calculate_expend(data, delay, maxd = 20):
    '''
    使用Cao虚假近邻算法 求解时序的扩展m
    --------------------------------------------
    :param data (series): 时序目标序列的原始数据
    :param delay (int) : 混沌时间序列的时延
    :param maxd (int)  : 最大扩展维度
    :param norm (string) : 距离计算使用的范数
    '''
#    tmp_root_path = 'tmp/distance/' + str(data.iloc[0])   
#    if tmp_root_path != None:
#        if not os.path.exists(tmp_root_path):
#            os.makedirs(tmp_root_path)
    
    sdata = pd.DataFrame(data)
    # E(d) = (n-dτ)^-1 * Σa(i,d)
    E = np.zeros(maxd)
    # E1 =  E(d+1) / E(d)
    E1 = np.zeros(maxd-1)
    for i in range(1,maxd+1):
        print("current dimension : ", i)
        psr_d,labels_d = psr(sdata, i, delay, sdata.columns[0])
        psr_dp1,labels_dp1 = psr(sdata, i+1, delay, sdata.columns[0])
        psr_d = psr_d[labels_d]
        psr_dp1 = psr_dp1[labels_dp1]
        # 取值范围为 1 - N-d*t
        irange = len(psr_dp1)
        psr_d = psr_d[:irange]
        # 求解d维数据的最近邻节点
        nnindex=calculate_nearest_node(psr_d, 'max')
        #a(i,d) = ||y_i(d+1) - y_n(i,d)(d+1)||∞ / ||y_i(d+1) - y_n(i,d)(d+1)||∞
        a = np.zeros(irange)
        
        tdiff_d = np.zeros(psr_d.shape)
        tdiff_dp1 = np.zeros(psr_dp1.shape)
        for j in range(irange):
            tdiff_d[j] = psr_d.iloc[j,:] - psr_d.iloc[int(nnindex[j]),:] + 1e-18    
            tdiff_dp1[j] = psr_dp1.iloc[j,:] - psr_dp1.iloc[int(nnindex[j]),:] + 1e-18
        tmaxnorm_d = maximum_norm(tdiff_d)
        tmaxnorm_dp1 = maximum_norm(tdiff_dp1)
        a = tmaxnorm_dp1/tmaxnorm_d
        E[i-1] = np.mean(a)
    E[0] = 100    # 让E1[1] -> 0;
    for i in range(maxd-1):
        E1[i] = E[i+1] / E[i]
#    plt.figure(figsize=(8,4),dpi=128)
#    plt.plot(np.arange(1,len(E1)+1),E1)
#    plt.title("dimension")
    return E1

def calculate_multivariate_expend_demension(data, labels, taolist, maxd = 4):
    '''
    多变量的嵌入维度计算
    使用[63] 卢山,王海燕.多变量时间序列最大李雅普诺夫指数的计算[J].物理学报,
    2006(02):572-576.中的改进PSR方法进行计算
    :TODO 由于近邻点的计算使用的是暴力方法，所以数据集不宜过大
    :TODO 输入特征的维度大小不易过大，因为扩展的维度最多将会有maxd^n个 n为特征维度。
    :TODO 代码需要整理， 包括变量的命名和循环的优化
    --------------------------------------------------------
    : param data (DataFrame): 包含多变量的数据集
    : param labels (list)   : 特征数据的标签
    : param taolist (list)  : 各变量对应的时延数组
    : param maxd (int)      : 最大的扩展维度
    : return mlist (list)   : 各变量对应的扩展维度
    '''
    # 输入数据的标准化
    if isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        data = data[labels]
    maxtao = np.max(taolist)
    if data.shape[0] < maxtao * maxd:
        raise ValueError("Data Set count is less than max demension")
    if data.shape[1] != len(taolist):
        raise ValueError("Variables demension isn't equal to taolist size")
    # 嵌入维度数组的计算
    m_all = []
    mlist = np.zeros(len(taolist), dtype=int)
    generate_demension_union(m_all, mlist, 0, maxd)
    
    count_m = len(m_all)    # 嵌入维度排列组合的数量
    count_v = len(taolist)  # 变量的数量
    #1.计算变量E(m_1,...,m_M) 
    #高复杂度 需要优化(对n个样本集求maxd^n次 (n*n-1)个近邻节点 与d次相空间重构)
    E_v = []
    for i in range(count_m):
        # log 
        print("---the %d group of %d  ---"%(i+1, len(m_all)))
        print(m_all[i])        
        psr_d, psr_l = psr_multivars(data, m_all[i], taolist, labels)
        psr_d = pd.DataFrame(psr_d[psr_l])
        # 节点重构过程
        psr_dp1s = []
        min_size = len(data)
        
        for j in range(count_v):
            _m = m_all[i].copy()
            _m[j] = _m[j]+1
            _p = []
            _p, _l = psr_multivars(data, _m, taolist, labels)
            _p = pd.DataFrame(_p[_l])
            if(len(_p) < min_size):
                min_size = len(_p)
            psr_dp1s.append(_p)
        
        psr_d = psr_d.iloc[:min_size]
        for j in range(len(psr_dp1s)):
            psr_dp1s[j] = psr_dp1s[j].iloc[:min_size]
        # 近邻点的计算
        nnindex = calculate_nearest_node(psr_d, 'max') 
        
        _diff = np.zeros(psr_d.shape)
        # 求E
        for j in range(min_size):
            _diff[j] = psr_d.iloc[j,:] - psr_d.iloc[int(nnindex[j]),:] + 1e-18
        tmaxnorm_d = maximum_norm(_diff)
        
        tmaxnorm_dp1s = np.zeros(psr_d.shape) # psr_d.shape == (min_size, len(psr_dp1s))
        for k in range(len(psr_dp1s)):
            psr_dp1 = psr_dp1s[k]
            _diff = np.zeros(psr_dp1.shape)
            for j in range(min_size):
                _diff[j] = psr_dp1.iloc[j,:] - psr_dp1.iloc[int(nnindex[j]),:] + 1e-18
            tmaxnorm_dp1s[:,k] = maximum_norm(_diff)
        tmaxnorm_dp1mean = np.array(tmaxnorm_dp1s).mean(axis=1) # 取扩展维度的均值
        e = tmaxnorm_dp1mean / tmaxnorm_d
        res = {}
        res['demension'] = m_all[i]
        res['value'] = np.mean(e)
        E_v.append(res)
    
    #2. 计算变量U(m) 
    U = dict() # U是存储字典的字典
    for i in range(count_m):
        m = np.sum(m_all[i])
        U[m] = []
    for _E in E_v:
        d_sum= np.sum(_E['demension'])
        U[d_sum].append(_E)
    #3. 计算变量E(m)
    E = dict()
    for key in U.keys():
        v_sum = 0
        for e in U[key]:
            v_sum = v_sum + e['value']
        v_mean = v_sum / len(U[key])
        E[key] = v_mean
    # 4. 最后计算E1
    E1 = dict()
    for m in U.keys():
        if m+1 in U.keys():
            E1[m] = E[m+1] / E[m]
    
    return E1, U # 先返回E1和U用于人工判别嵌入维度
#    # 确定最优m
#    suitM = 0
#    for m in E1.keys():
#        if E1[m] > 0.98:
#            suitM = m
#            break
#    # 返回
#    mlist = []
#    minE = 1e10
#    for _E in U[suitM  ]:
#        if minE > _E['value']:
#            minE = _E['value']
#            mlist = _E['demension']
#    
#    return mlist


def generate_demension_union(union, mlist, i, maxd):
    '''
    递归生成变量扩展维度的集合
    '''
    if(i == len(mlist)):
        union.append(mlist.copy())
    else:
        for num in range(1, maxd+1):
            mlist[i] = num
            generate_demension_union(union, mlist, i+1, maxd)
    
        
    
def calculate_nearest_node(data, norm='max'):
    '''
    计算节点链表中的最近节点(当前用的是暴力求解，待优化) 计算欧拉距离
    -----------------------------------------
    :param data (DataFrame) : 节点集, 列代表维度，行代表节点数量
    :param norm_func : 范数的计算函数
    :retrun nnindex(array) : 每个节点的最近节点下标， 为1维数组
    '''
    tmp = np.array(data)
    nnindex = np.zeros(len(tmp))
    for i in range(len(tmp)):
        nd = 2 * 1e10
#        if i == 0:
#            nd = calculate_dist(tmp[i],tmp[1], norm)
#            nnindex[i] = 1
#        else :
#            nd =  calculate_dist(tmp[i],tmp[0], norm)
#            nnindex[i] = 0
        for j in range(len(tmp)):
            if i != j:
                dist = calculate_dist(tmp[i],tmp[j], norm)
                if dist == 0 : # 去除近邻点是重合点的情况
                    continue
                if nd > dist:
                  nd = dist
                  nnindex[i] = j
    return nnindex

def calculate_dist(v1, v2, norm='two'):
    '''
    求解两个向量的距离
    ------------------------------------
    :param v1 (list) : 向量1
    :param v1 (list) : 向量2
    :param norm (string) : 计算的范数{"two", "max"}
    :return dist(float) : 返回两个向量的距离
    '''
    if norm == 'two':
        return np.sum(np.power((v1-v2),2)) # 没有开方减少计算复杂度
    if norm == 'max':
        return np.max(v1-v2)

def maximum_norm(data):
    '''
    对r各c为向量求解最大模范数
    ---------------------------------
    :param data (DataFrame) : 二维数组，存储r个c维向量
    :return maxnorm(array) : 返回每个元素的最大模范数
    '''
    tmp = np.array(data)
    res = np.zeros(len(tmp))
    for i in range(len(tmp)):
        res[i] = np.max(np.abs(tmp[i]))
    return res


##############################################################################
def psr(data, m, tao, tlabel, time_label=None):
    '''
    相空间重构过程。 (phase space reconstruction)
    -------------------------------------------
    :param data (DataFrame) ：数据源
    :param m     (int)      ：扩展维度
    :param tao   (int)      : 时延
    :param tlabel (str)     : 时序序列的标签
    :param time_label   (str, default==None): 时间标签，默认为空，用于记录生成序列的时间
    注： m * tao > len(data) 
    --------------------------------------------
    :return psr_data : 一维数据将返回 DataFrame 
                     ：(数据的特征为日期偏移量)
    '''

    if m*tao > len(data):
        raise ValueError("越界，数据量太小，小于m * τ")
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        
    psr_data = pd.DataFrame()
    labels = []
    for i in range(m):
        label_name = 'd' + str(i*tao)
        tmp = data[tlabel][i*tao:len(data)-(m-1-i)*tao].copy()
        tmp.index = np.arange(len(tmp))
        psr_data[label_name] = tmp
        labels.append(label_name)
    if time_label != None:
        psr_data[time_label] = data[time_label].copy()
    return psr_data,labels
    
        
def psr_multivars(data, m_list, tao_list, tlabel_list, time_label=None, is_reverse=True):
    '''
    多维数据的相空间重构过程。 (phase space reconstruction)
    -------------------------------------------
    :param data (DataFrame) ：数据源
    :param m        (list)  ：扩展维度数组
    :param tao      (list)  : 时延数组
    :param tlabel   (list)  : 时序序列的标签数组
    :param time_label (str, default==None): 时间标签，默认为空，用于记录生成序列的时间
    :param is_reverse (bool, default==False): 是否逆向重构(选第i日的前m个数据进行重构), 默认为空
    注： m * tao > len(data) 
    --------------------------------------------
    :return psr_data : 重构后的多维因子
    '''
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    psr_data = dict()
    labels = []
    if(len(m_list) != len(tao_list) or len(m_list) != len(tlabel_list)):
        raise ValueError("重构参数维度不匹配")
    
    min_size = len(data)
    for var_i in range(len(m_list)):
        m = m_list[var_i]
        tao = tao_list[var_i]
        tlabel = tlabel_list[var_i]
        for i in range(m):
            label_name = tlabel + '_d' + str(i*tao)
            if is_reverse == False:
                tmp = data[tlabel][i*tao:len(data)-(m-1-i)*tao].copy()
                tmp.index = np.arange(len(tmp))
            else:
                tmp = data[tlabel][(m-1-i)*tao:len(data)-i*tao].copy()                
                tmp.index = np.arange((m-1)*tao,len(data))
            psr_data[label_name] = tmp
            labels.append(label_name)
#         剔除空值 计算合法数据长度
        if(len(data)-(m-1)*tao < min_size):
            min_size = len(data) - (m-1)*tao 
#    if is_reverse == False:
#        psr_data = psr_data[:min_size]
#    else:
#        psr_data = psr_data[len(psr_data)-min_size:]      
#    psr_data.index = np.arange(len(psr_data))
#    if time_label != None:
#        if is_reverse == False:
#            psr_data[time_label] = data[time_label].copy()   
#        else:
#            psr_data[time_label] = np.array(data[time_label][len(data)-min_size:])
    if time_label != None :
        psr_data[time_label] = data[time_label].copy()   
    result = pd.DataFrame()        
    for k in psr_data.keys():
        result[k] = psr_data[k]
    if is_reverse == False:
        result = result[:min_size]        
    else:
        result = result[len(result)-min_size:]    
    result.index = range(len(result))
    return result, labels
    
#%% 测试
if __name__ == '__main__':
    
    ##################### 数据定义###################
    data_path = '../../data/IndexData/DJIA.csv'
    time_label = 'Ntime'
    ClosePrice = 'ClosePrice'
    ProfitRate = 'ProfitRate'
    value_label = ProfitRate

    data = pd.read_csv(data_path,usecols=[time_label, ClosePrice])
    import StockPredictionSystem as sps
    data[ProfitRate] = sps.difference_process_log(data[ClosePrice])
    data = data[1:]
    data.index = np.arange(len(data))
    # 去噪
    data = sps.denoise_method(data, value_label)
    
    ##################计算过程##################
    #计算时延
    taolist = calculate_delay(data[value_label], 30) 
    #计算扩增维度
    E = mlist = calculate_expend(data[value_label].iloc[:300], 6, 20)
    
    psr_data,labels = psr(data,m=3,tao=6,tlabel=value_label, time = time_label)
    
    ################ 可视化过程 ###############################
    import DataVisualization as dv
    dv.paint3Dscatter(psr_data[labels], ['x','y','z'])
    