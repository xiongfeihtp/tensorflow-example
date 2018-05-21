'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: MyBiRNN.py
@time: 2018/4/11 下午7:34
@desc: shanghaijiaotong university
'''
import tensorflow as tf
from tensorflow.contrib import rnn

n_steps = 28
n_input = 28
n_classes = 10
n_hidden = 300
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal(shape=[2 * n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal(shape=[n_classes]))

def BiRNN(x, weight, biases):
    x = tf.transpose(x, [1, 0, 2])  #以什么为分割标准，就放在第一个
    x = tf.reshape(x, [-1, n_input]) #
    x = tf.split(x, n_steps)  # axis = 0
    # 这样分割后，x->[n_steps, batch_size, n_input]
    #针对stack_bidireactional_rnn的特别输入
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, _, _ = rnn.stack_bidirectional_rnn(lstm_bw_cell, lstm_fw_cell, x, dtype=tf.float32)
    #output [n_steps, batch_size, 2 * hidden_size] 这里取最后一个时刻的状态作为
    return tf.matmul(outputs[-1], weight) + biases

"""
    rnn.stack_bidirectional_dynamic_rnn()
"""
rnn.stack_bidirectional_dynamic_rnn()

