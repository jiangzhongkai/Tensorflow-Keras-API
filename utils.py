"""-*- coding: utf-8 -*-
 DateTime   : 2019/4/30 12:07
 Author  : Peter_Bonnie
 FileName    : utils.py
 Software: PyCharm
"""

from __future__ import absolute_import,division,print_function
import time
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score,precision_score,accuracy_score
import shutil
import datetime
import glob
import pickle

#TODO:function tools

def normalization(data,min,max):

    return (data-min)/(max-min)

def standarlization(data,std,mean):

    return (data-mean)/std

def reconstruct(data,std,mean):

    return data*std+mean


def compute_runing_time(diff_time):
    """computer run time

    Args:
        start(time)
        end(time)

    Returns:
        hours,mins,secs
    """
    hours,seconds=divmod(diff_time,3600)
    mins,secs=divmod(seconds,60)

    return "it costs:{0}小时:{1}分钟:{2}秒".format(hours,mins,secs)

def R(target,pred):
    """compute R^2"""

    assert len(target)==len(pred)
    SSE = sum((i - j) ** 2 for i, j in zip(target, pred))
    VAR = sum((i - target.mean()) ** 2 for i in target)

    return 1-SSE/VAR

def RMSE(true,pred):

    assert len(true)==len(pred)

    return np.sqrt(np.mean(np.sum([np.square(i-j) for i,j in zip(true,pred)])))


def MSE(true,pred):

    assert len(true)==len(pred)

    return np.mean(np.sum([np.square(i-j) for i,j in zip(true,pred)]))

def MAPE(true,pred):

    assert len(true)==len(pred)

    return np.mean(np.sum([np.abs(i-j)/i*1.0 for i,j in zip(true,pred)]))

def MAE(true,pred):

    assert len(true)==len(pred)

    return np.mean(np.sum([np.abs(i-j) for i,j in zip(true,pred)]))

def Precision(true,pred):
    """get precision score

    Args:
        true(np array): true value
        pred(np array): predict value

    Returns:
        precision score(float)

    """
    return precision_score(true,pred)

def Recall(true,pred):
    """get recall score

    Args:
        true(np array): true value
        pred(np array): predict value

    Returns:
        recall score(float)

    """
    return recall_score(true,pred)

def F_score(beta,true,pred):
    """set different value for beta

    Args:
        beta(float): if beta<1.0,the weight of precision higher than the recall,otherwise,the recall weight is higher than the precision
        true(np array): the true value
        pred(np array): the predict value

    Returns:
        result

    """
    return (1+beta**2)*Precision(true,pred)*Recall(true,pred)/(beta**2*Precision(true,pred)+Recall(true,pred))


def reshape_data(data,newshape):
    """reshape data
    Args:
        data(file obj):source file
        newshape(list or tuple): target shape

    Returns:
        data wih reshaped

    Raise:
       file not found

    """
    try:
        data=np.loadtxt(data)
    except:
        raise FileNotFoundError("{} file not found, please check the input file path is exist or not.".format(data))
    else:
        data=np.reshape(data,newshape=newshape)

        return data

def main():

    #TODO:归一化操作
    data=pd.read_csv("data/power_8.csv")
    data=data.values
    _mean=data.mean()
    _std=data.std()
    data=standarlization(data,mean=_mean,std=_std)
    pd.DataFrame(data=data,columns={'useage'}).to_csv('data/power_pro_8.csv',index=False)

    #TODO:还原操作
    #...



if __name__=="__main__":
    main()