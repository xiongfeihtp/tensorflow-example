'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: Mytest.py
@time: 2018/4/12 上午11:16
@desc: shanghaijiaotong university
'''
import tensorflow as tf
import re
from collections import Counter
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

stop_words_path = './chinese_stopword.txt'

def get_stop_word(file):
    with open(file, 'r') as f:
        words = f.readlines()
        words_dict = {word.strip(): word.strip() for word in words}
    return words_dict

def pad_sentence(sentence, padding_word="<PAD/>", sequence_length=400):
    if sequence_length > len(sentence):
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
    else:
        new_sentence = sentence[:sequence_length]
    return new_sentence

def clean_line(line, stop_words):
    line = re.sub(r'[0-9|\u3000]', '', line)
    words = line.split(' ')
    filtered_words = [word for word in words[:-1] if word not in stop_words]
    return filtered_words, words[-1]


def data_label_load(filename):
    data_list = []
    y_list = []
    with open(filename, 'r') as f:
        stop_words = get_stop_word(stop_words_path)
        for line in f:
            line, label = clean_line(line, stop_words)
            data_list.append(line)
            y_list.append(label)
    return data_list, y_list


def get_vocabuary(data_list):
    word_count = Counter()
    for line in data_list:
        word_count.update(line)
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    WordToindex = {item[0]: i for i, item in enumerate(sorted_words, start=2)}
    indexToWord = {i: item[0] for i, item in enumerate(sorted_words, start=2)}
    WordToindex["<PAD/>"] = 0
    WordToindex["<Ukn/>"] = 1
    indexToWord[0] = "<PAD/>"
    indexToWord[1] = "<Ukn/>"
    print("vocabuary_size: {}".format(len(WordToindex)))
    return WordToindex, indexToWord


def build_example(datas, labels):
    # 生成examle格式字典
    examples = []
    for data, label in zip(datas, labels):
        example = {"text": data, "label": label}
        examples.append(example)
    return examples


def batch_iter(data, label, batch_size, num_steps, shuffle=True):
    data_size = len(label)
    num_batches_per_epoch = data_size // batch_size
    num_epoches = num_steps // num_batches_per_epoch
    for epoch in range(num_epoches):
        if shuffle:
            shuffle_index = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_index]
            shuffle_label = label[shuffle_index]
        else:
            shuffle_data = data
            shuffle_label = label
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index: end_index], shuffle_label[start_index:end_index]


def build_features(config, examples, data_type, out_file, word2idx_dict):
    # 设置元素作为序列的最大长度，便于对齐
    para_limit = config.sequence_len
    # 生成tfrecord file
    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    for example in tqdm(examples):
        # 向量
        # 检查数据格式是否正确，避免和这次一样的错误
        raw_text = pad_sentence(example['text'], sequence_length=para_limit)
        #这里一定要设置dtype类型，转换前和转换后一定要一致，否则就会出现数据错误
        text = np.array([word2idx_dict[word] for word in raw_text],dtype=np.int32)
        label = np.array(example['label'], np.int32)
        #如何写入序列化向量和矩阵，全部转化为BytesList
        record = tf.train.Example(features=tf.train.Features(feature={
            "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.tostring()])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
        }))
        writer.write(record.SerializeToString())

def prepro(config):
    train_data, train_label = data_label_load(config.train_data_path)
    test_data, test_label = data_label_load(config.test_data_path)
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(len(test_data)))
    # label encode
    le = preprocessing.LabelEncoder()
    le.fit(train_label + test_label)
    train_label = le.transform(train_label)
    test_label = le.transform(test_label)
    train_examples = build_example(train_data, train_label)
    test_examples = build_example(test_data, test_label)
    WordToindex, indexToWord = get_vocabuary(train_data + test_data)
    build_features(config, train_examples, "train", config.train_record_file, WordToindex)
    build_features(config, test_examples, "test", config.test_record_file, WordToindex)
