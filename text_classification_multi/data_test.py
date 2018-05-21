'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: test.py
@time: 2018/4/14 上午12:01
@desc: shanghaijiaotong university
'''
import tensorflow as tf
from graph_handler import GraphHandler
from model import Model
from tqdm import tqdm
import os
import numpy as np
from util import *


def get_record_parser():
    def parse(example):
        para_limit = 400
        features = tf.parse_single_example(example,
                                           features={
                                               "text": tf.FixedLenFeature([], tf.string),
                                               "label": tf.FixedLenFeature([], tf.string)
                                           })
        # 因为设置了固定的对齐长度，所以可以直接进行reshape恢复成正常的tensor
        text = tf.decode_raw(
            features["text"], tf.int32)
        label = tf.decode_raw(
            features["label"],tf.int32)
        return text, label
    return parse


# 训练数据迭代器
def get_batch_dataset(record_file, parser):
    num_threads = tf.constant(4, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(1000).repeat()
    dataset = dataset.batch(5)
    return dataset

print("Building model...")
parser = get_record_parser()

train_dataset = get_batch_dataset('./data/train.tfrecords', parser)
# 创建一个迭代器
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, train_dataset.output_types, train_dataset.output_shapes)

train_iterator = train_dataset.make_one_shot_iterator()

text, label = iterator.get_next()
# 实例化一个迭代器
with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())
    for _ in range(100):
        text_val, label_val = sess.run([text, label], feed_dict={handle: train_handle})
        print(text_val.shape)
        print(text_val[0])
        print(label_val.shape)
        print(label_val)
