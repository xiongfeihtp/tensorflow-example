'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: Myword2vec.py
@time: 2018/4/9 下午10:12
@desc: shanghaijiaotong university
'''
import tensorflow as tf
from collections import Counter, deque
import numpy as np
import math
from tqdm import tqdm
import random

words = Counter()
with open('news_fasttext_test.txt', 'r') as f:
    raw_data = f.readlines()
    for line in tqdm(raw_data):
        words.update(line.split()[:-1])

reverse_dictionary = {}
data = []
data_index = 0


# 队列来实现skip_gram样本生成，建立队列实现滑窗
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # 这里为什么labels设为(batch_size, 1)
    span = 2 * skip_window + 1
    buffer = deque(maxlen=span)
    # 循环
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        #无放回式抽取
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        #队列来实现采样词区间
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


vocabulary_size = 10000

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100

valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # replace表示复用
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # 这里为什么labels设为[batch_size, 1]，nce_loss target [batch_size, num_true] num_true默认为1
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # tensorflow中的embedding表示法
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_normal([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # 负采样频率，弄懂这个函数
    # 和embeddings矩阵相同，要建立vocabulary_size个logistics regression，这里需要初始化所有的weight和bias
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # 计算负采样loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize()
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()

    num_steps = 100001

    with tf.Session(graph=graph) as sess:
        init.run()
        print("Initialized")
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("average_loss as step", step, ':', average_loss)
                average_loss = 0
            if step % 10000 == 0:
                # 另外一种调用
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1] #排除自己
                    pass


