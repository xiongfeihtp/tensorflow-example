'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: config.py
@time: 2018/4/13 下午2:07
@desc: shanghaijiaotong university
'''
import os
import tensorflow as tf

from train import train
from data_preprocess import prepro

flags = tf.flags

flags.DEFINE_string("train_data_path" ,"./mini_data_train.txt", "train_data_path")
flags.DEFINE_string("test_data_path" ,"./mini_data_test.txt", "test_data_path")

flags.DEFINE_string("run_id", "0", "RUN ID[0]")
flags.DEFINE_string("model_name", "basic", "model name")

target_dir = "data"
log_dir = os.path.join(flags.FLAGS.model_name, flags.FLAGS.run_id, "event")
save_dir = os.path.join(flags.FLAGS.model_name, flags.FLAGS.run_id, "save")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")

flags.DEFINE_string("train_record_file" ,"./data/train.tfrecords", "train_tf_file")
flags.DEFINE_string("test_record_file" ,"./data/test.tfrecords", "test_tf_file")


flags.DEFINE_string("load_path", None, "retrain_path")
flags.DEFINE_integer("load_step", 0, "retrain globel step")

flags.DEFINE_integer("vocabuary_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("embedding_dim", 300, "Embedding dimension for Glove")

flags.DEFINE_integer("sequence_len", 400, "Limit length for text")
flags.DEFINE_boolean("use_cudnn", False, "Whether to use cudnn rnn (should be False for CPU)")

flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")

flags.DEFINE_integer("val_num_batches", 1, "Number of batches to evaluate the model")

flags.DEFINE_float("init_lr", 0.5, "Initial learning rate for Adadelta")
flags.DEFINE_float("keep_prob", 0.7, "Dropout keep prob in rnn")


flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden_size", 75, "Hidden size")
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")
flags.DEFINE_integer("max_to_keep", 10, "max numbers of saved model")
flags.DEFINE_integer("num_class", 14, "number of classes")
flags.DEFINE_integer("test_nums", 10, "number of test samples")

flags.DEFINE_integer("num_threads", 4, "data_load")
flags.DEFINE_integer("capacity", 15000, "data_load_shuffle_capacity")
def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    else:
        print("Unknown mode")
        exit(0)
if __name__ == "__main__":
    tf.app.run()
