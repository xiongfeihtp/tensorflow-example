'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: tensorflow_code_structure.py
@time: 2018/4/12 上午10:13
@desc: shanghaijiaotong university
'''
import functools
import tensorflow as tf
#定义子图模块，并且仅在函数调用的第一次进行建图，并且给这一部分进行scope封装
def define_scope(function):
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator
#使用方法
class Model:
    def __init__(self, data, target):
        self.data= data
        self.target = target
        self.prediction = None
        self.optimize = None
        self.error = None
    @define_scope
    def train(self):
        pass

    @define_scope
    def prediction(self):
        pass

    @define_scope
    def error(self):
        pass




