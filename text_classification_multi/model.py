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


class Model:
    def __init__(self, config, batch, trainable=True, opt=True):
        self.config = config
        # 通用的配置在cpu上设置
        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            # 通过这个变量控制训练还是interfence
            self.is_training = tf.get_variable("is_training", shape=[], dtype=tf.bool, trainable=False)
            # 这种方式比较耗内存和慢的，因为在graph中加入了constant节点，这里采取导入预训练词向量的方式，可以通过placeholder进行传入
            if trainable:
                self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
                # self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
                # 这里单机gpu并行Adadelta会出现错误
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        if trainable:
            tower_grads = []
            for i in range(config.num_gpus):
                with tf.device('/gpu:{}'.format(i)):
                    with tf.variable_scope("{}_{}".format("gpu", i)) as scope:
                        self.loss = self.tower(batch, config, opt, scope)
                        #在每个gpu上定义完成共享的模型
                        tf.get_variable_scope().reuse_variables()
                        grads = self.optimizer.compute_gradients(self.loss)
                        gradients, tvars = zip(*grads)
                        clipped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                        grads = list(zip(clipped_grads, tvars))
                        # embedding层的参数跟新会出现问题，因为不同的batch跟新的参数会部分不同，需要筛选
                        tower_grads.append(grads)
            grads = self.average_gradients(tower_grads)
            self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)
        else:
            with tf.name_scope("test") as scope:
                self.loss = self.inference(scope)

    def tower(self, batch, config, opt, scope):
        self.input, self.labels = batch.get_next()
        self.mask = tf.cast(self.input, tf.bool)
        self.len = tf.reduce_mean(tf.cast(self.mask, tf.int32), axis=1)
        if opt:
            batch_size = config.batch_size
            self.max_len = tf.reduce_max(tf.cast(self.mask, tf.int32))
            self.input = tf.slice(self.input, [0, 0], [batch_size, self.max_len])
            self.mask = tf.slice(self.mask, [0, 0], [batch_size, self.max_len])
        else:
            self.max_len = config.sequence_len
        return self.inference(scope)

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            v = grad_and_vars[0][1]
            average_grads.append((grad, v))
        return average_grads

    def inference(self, scope):
        config = self.config
        gru = cudnn_gru if config.use_cudnn else native_gru
        #这里将embedding matrix存放在各个gpu上，加快计算效率
        self.word_mat = tf.get_variable("word_mat", initializer=tf.truncated_normal(
            shape=[config.vocabuary_size, config.embedding_dim]), trainable=True)
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
        with tf.variable_scope("predict"):
            logits = slim.fully_connected(output, config.num_class, activation_fn=None)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
            losses = tf.get_collection('losses', scope)
            total_loss = tf.add_n(losses, name="total_loss")
            return total_loss
