'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: train.py
@time: 2018/4/13 上午11:42
@desc: shanghaijiaotong university
'''
import tensorflow as tf
from graph_handler import GraphHandler
from model import Model
from tqdm import tqdm
import os
import numpy as np
from util import *

def train(config):
    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    test_dataset = get_dataset(config.test_record_file, parser, config)
    # 创建一个迭代器
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    #实例化一个迭代器
    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()
    model = Model(config, iterator)
    graph_handler = GraphHandler(config, model)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    loss_save = 100.0
    patience = 0
    lr = config.init_lr
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        # 导入预训练模型
        graph_handler.initialize(sess)
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        sess.run(tf.assign(model.is_training, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
        for _ in tqdm(range(config.num_steps)):
            global_step = sess.run(model.global_step) + 1
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
            if global_step % config.period == 0:
                print("num_step:{} train_loss:{}".format(global_step, loss))
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                graph_handler.add_summary(loss_sum, global_step)
            if global_step % config.checkpoint == 0:
                print("test on {}".format(global_step))
                sess.run(tf.assign(model.is_training,
                                   tf.constant(False, dtype=tf.bool)))
                """
                eval
                """
                losses = []
                for _ in tqdm(range(config.test_nums // config.val_num_batches)):
                    test_loss = sess.run([model.loss], feed_dict={handle: test_handle})
                    losses.append(test_loss)
                test_loss = np.mean(losses)
                test_loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="eval/loss", simple_value=test_loss), ])
                graph_handler.add_summary(test_loss_sum, global_step)
                print("num_step:{} test_loss:{}".format(global_step, test_loss))
                if test_loss < loss_save:
                    loss_save = test_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = test_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                sess.run(tf.assign(model.is_training,
                                   tf.constant(True, dtype=tf.bool)))
                # 特殊的学习率变化规律
                graph_handler.writer.flush()
                filename = os.path.join(
                    config.save_dir, "{}_{}.ckpt".format(config.model_name, global_step))
                graph_handler.save(sess, filename)
