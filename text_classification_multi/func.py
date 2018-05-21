'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: func.py
@time: 2018/4/12 下午10:44
@desc: shanghaijiaotong university
'''
import tensorflow as tf
from tensorflow.contrib import cudnn_rnn
from tensorflow.contrib import rnn


def dropout(args, keep_prob, is_train, mode="rnn"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "rnn" and len(args.get_shape().to_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
            # 对dropout进行noise_shape控制，并且*scale不进放大，默认情况下需要进行放大
        args = tf.cond(is_train, lambda: tf.nn.dropout(args, keep_prob, noise_shape) * scale, lambda: args)
    return args

# 定义自己的rnn变体
# 实现__init__ 和 __call__
class cudnn_gru:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=True, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.params = []
        self.inits = []
        self.dropout_masks = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units, input_size=input_size_)
            gru_bw = cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units, input_size=input_size_)
            # 初始化参数
            # 通过get_variable 和 variable_scope 可以进行参数共享，实现多GPU共同更新一个模型的梯度，实现多batch
            param_fw = tf.get_variable(name="params_fw{}".format(layer),
                                       initializer=tf.random_uniform([gru_fw.params_size()], -0.1, 0.1),
                                       validate_shape=False)
            param_bw = tf.get_variable(name="params_bw{}".format(layer),
                                       initializer=tf.random_uniform([gru_bw.params_size()], -0.1, 0.1),
                                       validate_shape=False)
            init_fw = tf.get_variable(name="init_fw{}".format(layer), initializer=tf.zeros([1, batch_size, num_units]),
                                      trainable=False)
            init_bw = tf.get_variable(name="init_bw{}".format(layer), initializer=tf.zeros([1, batch_size, num_units]),
                                      trainable=False)
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32), keep_prob=keep_prob,
                              is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32), keep_prob=keep_prob,
                              is_train=is_train, mode=None)
            self.grus.append([gru_fw, gru_bw])
            self.params.append([param_fw, param_bw])
            self.inits.append([init_fw, init_bw])
            self.dropout_masks.append([mask_fw, mask_bw])

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layer=False):
        """
        :param inputs:
        :param seq_len: 这里seq_len保证了reverse时，填充的元素不会反序
        :param keep_prob:
        :param is_train:
        :param concat_layer:
        :return:
        """
        # 针对cudnn接口的固定模式[sequence_len, batch_size, dim]
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            param_fw, param_bw = self.params[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_masks[layer]
            with tf.variable_scope("fw"):
                output_fw, _ = gru_fw(outputs[-1] * mask_fw, init_fw, param_fw)
            with tf.variable_scope("bw"):
                input_bw = tf.reverse_sequence(outputs[-1], seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                output_bw, _ = gru_bw(input_bw * mask_bw, init_bw, param_bw)
                output_bw = tf.reverse_sequence(output_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([output_fw, output_bw], axis=2))
        if concat_layer:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=True, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_masks = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = rnn.GRUCell(num_units)
            gru_bw = rnn.GRUCell(num_units)
            init_fw = tf.Variable(tf.zeros([batch_size, num_units]))
            init_bw = tf.Variable(tf.zeros([batch_size, num_units]))
            mask_fw = dropout(tf.ones(shape=[batch_size, 1, input_size_], dtype=tf.float32), keep_prob=keep_prob,
                              is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones(shape=[batch_size, 1, input_size_], dtype=tf.float32), keep_prob=keep_prob,
                              is_train=is_train, mode=None)
            self.grus.append([gru_fw, gru_bw])
            self.inits.append([init_fw, init_bw])
            self.dropout_masks.append([mask_fw, mask_bw])

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=True, concat_layer=False):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_masks[layer]
                with tf.variable_scope('fw_{}'.format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(gru_fw, outputs[-1] * mask_fw, sequence_length=seq_len,
                                                  initial_state=init_fw, dtype=tf.float32)

                with tf.variable_scope('bw_{}'.format(layer)):
                    input_bw = tf.reverse_sequence(outputs[-1], seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(gru_bw, input_bw * mask_bw, sequence_length=seq_len,
                                                  initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layer:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


class cudnn_gru_raw:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.params = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_)
            param_fw = tf.Variable(tf.random_uniform(
                [gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
            param_bw = tf.Variable(tf.random_uniform(
                [gru_bw.params_size()], -0.1, 0.1), validate_shape=False)
            init_fw = tf.Variable(tf.zeros([1, batch_size, num_units]))
            init_bw = tf.Variable(tf.zeros([1, batch_size, num_units]))

            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw,))
            self.params.append((param_fw, param_bw,))
            self.inits.append((init_fw, init_bw,))
            self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=False):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            param_fw, param_bw = self.params[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw"):
                out_fw, _ = gru_fw(outputs[-1] * mask_fw, init_fw, param_fw)
            with tf.variable_scope("bw"):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, init_bw, param_bw)
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        # 防止退化，这样的维度加深？
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res
