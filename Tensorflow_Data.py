"""-*- coding: utf-8 -*-
 DateTime   : 2019/5/15 20:54
 Author  : Peter_Bonnie
 FileName    : Tensorflow_Data
 Software: PyCharm
"""
"""
mainly used for data slices,load,and batch feed into custom model by using tensorflow method.  
"""

import tensorflow as tf
from tensorflow.python.ops import array_ops,math_ops,init_ops
from typing import Tuple,List
import numpy as np
import os
import pandas as pd

tf.enable_eager_execution()

class DATA(object):

    def __init__(self,**kwargs):

        super(self,DATA).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        pass

    def set_window_to_get_data(self,size,data_series,target_series,df=pd.DataFrame()):
        """get x and y data by setting window

        Args:
            size(integer):window size
            data_series:multi feature series of input data
            target_series: it is target feature that we aim to predict

        Returns:
            Array:x_data,y_data
        """
        X=df[data_series].values
        Y=df[target_series].values

        X_DATA=[]
        Y_DATA=[]

        for i in range(len(X)-size+1):
            X_DATA.append(X[i:i+size])
            Y_DATA.append(Y[i:i+size])

        return np.array(X_DATA),np.array(Y_DATA)

    def get_np_dataset(self,config,cat_before_window,default_file):
        """ """
        X_DATA=''
        Y_DATA=''
        return X_DATA,Y_DATA


    def get_dataset(self,config,shuffled=True,default_file=None,use_validate=True):

        """split data set into train data, test data and validate data
        Ae


        """

        X_DATA,Y_DATA=self.get_np_dataset(config)
        test_size=537

        if use_validate:
            global val_size
            val_size=test_size
            train_size=len(X_DATA)-test_size*2
        else:
            train_size=len(X_DATA)-test_size

        #read x_data and y_data from orignal data
        dataset=tf.data.Dataset.zip(
            (
             tf.data.Dataset.from_tensor_slices(X_DATA),
             tf.data.Dataset.from_tensor_slices(Y_DATA),
            )
        )
        #get train data,validate/test data
        train_data=dataset.take(train_size)

        if shuffled:
            train_data=train_data.shuffle(buffer_size=train_size,reshuffle_each_iteration=True)

        if use_validate:
            validate_data=dataset.skip(train_size).take(val_size)
            test_data=dataset.skip(train_size+val_size).take(test_size)
            return train_data,validate_data,test_data
        else:
            test_data=dataset.skip(train_size).take(test_size)

        return train_data,test_data

    def make_one_shot_iterator(self,config,default_file,use_validate):
        """get one shot iterator"""

        if use_validate:
            train_data,val_data,test_data=self.get_dataset(config=config,default_file=default_file,use_validate=use_validate)
            it=train_data.make_one_shot_iterator()
            lit=val_data.make_one_shot_iterator()
            tit=test_data.make_one_shot_iterator()

            return it,lit,tit
        else:
            train_data,test_data = self.get_dataset(config=config, default_file=default_file,
                                                               use_validate=use_validate)
            it = train_data.make_one_shot_iterator()
            tit = test_data.make_one_shot_iterator()
            return it,tit

    def get_batch_data(self,config,session,use_validate=True,default_file=None):

        """get batch size data to feed into our model

        Args:
            config:object of Config
            session: start a session

        Returns:
            batch_x_data,batch_y_data
        """
        if  not os.path.exists(default_file):
            raise FileNotFoundError("{0} not found".format(default_file))

        session.run(tf.global_variables_initializer())
        if use_validate:

            #step1:get x_data and y_data
            train_data,validate_data,test_data=self.get_dataset(config,default_file=default_file,use_validate=use_validate)

            #step2:if the rest sample number is less than batch_size,we drop the rest samples
            train_data=train_data.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))
            validate_data=validate_data.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))
            test_data=test_data.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))

            #step3:initial the iterator
            train_iterator=train_data.make_initializable_iterator()
            val_iterator=validate_data.make_initialzable_iterator()
            test_iterator=test_data.make_initialzable_iterator()

            #step4:get the next batch_size data by iterator method.
            train_next_element=train_iterator.get_next()
            validate_next_element=val_iterator.get_next()
            test_next_element=test_iterator.get_next()

            return train_next_element,validate_next_element,test_next_element
        else:
            train_data,test_data=self.get_dataset(config,default_file=default_file,use_validate=use_validate)

            train_data=train_data.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))
            test_data=test_data.apply(tf.contrib.data.batch_and_drop_remainder(config.batch_size))

            train_iterator=train_data.make_initializable_iterator()
            test_iterator=test_data.make_initialzable_iterator()

            train_next_element=train_iterator.get_next()
            test_next_element=test_iterator.get_next()

            return train_next_element,test_next_element










