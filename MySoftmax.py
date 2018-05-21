import tensorflow as tf
import numpy as np
train_X = np.random.rand(1000,785)
train_Y = np.random.randint(1,10,1000)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 785])
_y = tf.placeholder(tf.int32, [None])
W = tf.Variable(tf.zeros([785, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#tf.nn.sparse_softmax_cross_entropy_with_logits 只接受int类型的数据，并只支持one_hot类型的labels
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = _y, logits = y ))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
tf.global_variables_initializer().run()
#数据生成器的编写，shuffle功能
def batch_iter(X, y, batch_size, num_epochs, shuffle=True):
    data_nums = len(y)
    num_batches_per_epoch = int(data_nums / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_index = np.random.permutation(np.arange(data_nums))
            shuffed_X = X[shuffle_index]
            shuffed_y = y[shuffle_index]
        else:
            shuffed_X = X
            shuffed_y = y

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_nums)
            yield shuffed_X[start_index:end_index], shuffed_y[start_index:end_index]
global_step = 0
for batch_X, batch_y in batch_iter(train_X, train_Y, 32, 100):
    global_step += 1
    loss_val, _ = sess.run([loss, train_step], feed_dict = {x : batch_X, _y : batch_y})
    print("step: {}, loss: {}".format(global_step, loss_val))
