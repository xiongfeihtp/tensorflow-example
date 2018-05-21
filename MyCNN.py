import tensorflow as tf

batch_size = 32
image_height = 24
image_width = 24


# 带l2正则的W定义方式
def variable_with_l2(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
        tf.add_to_collection('losses', weight_loss)
    return var

def model():
    image_holder = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, shape=[batch_size])
    with tf.variable_scope("layer1") as scope:
        weight1 = variable_with_l2(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
        kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding="SAME")
        bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
        conv1 = tf.nn.bias_add(kernel1, bias1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    with tf.variable_scope("layer2") as scope:
        weight2 = variable_with_l2(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
        kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding="SAME")
        bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
        conv2 = tf.nn.bias_add(kernel2, bias2)
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    flatten = tf.reshape(pool2, [batch_size, -1])
    dim = flatten.get_shape()[-1].value
    with tf.variable_scope("layer3") as scope:
        weight3 = variable_with_l2(shape=[dim, 384], stddev=0.04, w1=0.004)
        bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
        local3 = tf.nn.relu(tf.matmul(flatten, weight3) + bias3)

    with tf.variable_scope("layer4") as scope:
        weight4 = variable_with_l2(shape=[384, 192], stddev=0.04, w1=0.004)
        bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
        local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
    with tf.variable_scope("logit") as scope:
        weight5 = variable_with_l2(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
        bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
        logit = tf.add(tf.matmul(local4, weight5), bias5)

    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label_holder)
        loss_logit = tf.reduce_mean(cross_entropy, name="cross_entropy")
        tf.add_to_collection('losses', loss_logit)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss)
    # 针对分类问题topk精度
    topk = tf.nn.in_top_k(predictions=logit, targets=label_holder, k=1)
    return train_op, topk

def data_generator():
    pass

