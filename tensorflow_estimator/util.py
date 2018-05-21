'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: util.py
@time: 2018/5/19 下午4:46
@desc: shanghaijiaotong university
'''
import tensorflow as tf
import os
import time

def list_files(input_data):
    if tf.gfile.IsDirectory(input_data):
        # 排除隐藏文件
        file_name = [f for f in tf.gfile.ListDirectory(input_data) if not f.startswith('.')]
        return [os.path.join(input_data, f) for f in file_name]
    else:
        return [input_data]


def elapse_time(start_time):
    return round((time.time() - start_time) / 60)

