'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: train.py
@time: 2018/4/12 下午1:15
@desc: shanghaijiaotong university
'''
import tensorflow as tf
from func import cudnn_gru, native_gru, dropout
from tensorflow.contrib import slim
from tensorflow.contrib.metrics import streaming_accuracy


class Model:
    def __init__(self, config, batch, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # 通过这个变量控制训练还是interfence
        self.is_training = tf.get_variable("is_training", shape=[], dtype=tf.bool, trainable=False)
        # 这种方式比较耗内存和慢的，因为在graph中加入了constant，这里采取导入预训练词向量的方式
        self.input, self.labels = batch.get_next()
        self.word_mat = tf.get_variable("word_mat", initializer=tf.truncated_normal(
            shape=[config.vocabuary_size, config.embedding_dim]), trainable=True)
        self.mask = tf.cast(self.input, tf.bool)
        self.len = tf.reduce_mean(tf.cast(self.mask, tf.int32), axis=1)
        if opt:
            batch_size = config.batch_size
            self.max_len = tf.reduce_max(tf.cast(self.mask, tf.int32))
            self.input = tf.slice(self.input, [0, 0], [batch_size, self.max_len])
            self.mask = tf.slice(self.mask, [0, 0], [batch_size, self.max_len])
        else:
            self.max_len = config.sequence_len

        self.inference()
        if trainable:
            self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, tvars = zip(*grads)
            clipped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=self.global_step)

        with tf.name_scope("accuracy"):
            self.accuracy, self.accuracy_op= streaming_accuracy(predictions=self.prediction, labels=self.labels)

    def inference(self):
        config = self.config
        gru = cudnn_gru if config.use_cudnn else native_gru
        with tf.variable_scope("embedding"):
            # [batch_size, sequence_len, embedding_size]
            emb_input = tf.nn.embedding_lookup(self.word_mat, self.input)
        emb_input = tf.reshape(emb_input, [config.batch_size, self.max_len, config.embedding_dim])
        with tf.variable_scope("encode") as sc:
            rnn = gru(num_layers=2,
                      num_units=config.hidden_size,
                      batch_size=config.batch_size,
                      input_size=emb_input.get_shape().as_list()[-1],  # 注意传入的shape参数形式
                      keep_prob=config.keep_prob,
                      is_train=self.is_training,
                      scope=sc)
            output = rnn(emb_input, seq_len=self.len)
        begin = tf.slice(output, [0, 0, 0], [-1, 1, -1])
        after = tf.slice(output, [0, self.max_len - 1, 0], [-1, 1, -1])
        output = tf.concat([begin, after], axis=2)
        with tf.variable_scope("prediction"):
            logits = slim.fully_connected(output, config.num_class, activation_fn=None)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
            self.prediction = tf.argmax(slim.softmax(logits), axis=1)
