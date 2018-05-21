'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: MyLSTM_language_Model.py
@time: 2018/4/11 下午6:50
@desc: shanghaijiaotong university
'''
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import framework

hidden_size = 300


def lstm_cell():
    return rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)


attn_cell = lstm_cell

if is_training and config.keep_prob < 1:
    # 包装drop out
    # 多层封装，必须这样定义子函数然后进行rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)])

    def attn_cell():
        return rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)


    # 层数叠加
    cell = rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)])

    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
    if is_training and config.keep_prob < 1:
        inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)

    outputs = []
    state = _initial_state  # 每一个batch之后初始状态会重置
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step:], state)
            outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, axis=1), [-1, hidden_size])

# 关于学习率和梯度的控制方法
_lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
optimizer = tf.train.GradientDescentOptimizer(_lr)
train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=framework.get_or_create_global_step())

# 对学习率进行赋值
_new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
_lr_update = tf.assign(_lr, _new_lr)

def assign_lr(sess, lr_value):
    sess.run(_lr_update, feed_dict={_new_lr: lr_value})

