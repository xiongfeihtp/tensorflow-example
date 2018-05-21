'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: Myresnet.py
@time: 2018/4/8 下午7:43
@desc: shanghaijiaotong university
'''

import collections
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import utils

"""
降采样，即是采样点数减少。对于一幅N*M的图像来说，如果降采样系数为k,则即是在原图中 
每行每列每隔k个点取一个点组成一幅图像。降采样很容易实现. (conv2d(inputs,[1, 1], stride=k, padding='SAME'))实现

升采样，也即插值。对于图像来说即是二维插值。如果升采样系数为k,即在原图n与n+1两点之间插入k-1个点，使其构成k分。
二维插值即在每行插完之后对于每列也进行插值。 
"""

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    pass

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs

    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

# 作用，实现subsample??
@slim.add_arg_scope
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, kernel_size, stride=1, padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)

@slim.add_arg_scope  # 只有被修饰过的函数才能使用slim.arg_scope
def stack_blocks_dense(net, blocks, output_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                # 这里没有创建了不同的scope
                with tf.variable_scope('unit_{}'.format(i + 1), values=[net]):
                    # 三层残差模块，这里循环中net会不停的被覆盖
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)

            net = utils.collect_named_outputs(output_collections, sc.name, net)
    return net

def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': weight_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        # xavier初始化方法
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # resnet_v2
        depin_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if depth == depin_in:  # depth是第三层的输出通道数
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = residual + shortcut
        return utils.collect_named_outputs(outputs_collections, sc.name, output)

def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=True,
              scope=None):
    # scope_name  default_name  variable
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        # 创建集合名
        end_points_collection = sc.original_name_scope + '_end_points'
        # 收集多个end_points的方法
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # reduce_mean实现全局池化
                # batch_size [height width] channels -> batch_size 1 1 channels
                # reduce_mean实现全局池化 [1, 2]
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                # 通过一维卷积代替全连接
            if num_classes is not None:
                # conv2d(inputs, num_classes, [1, 1], without activation and normalize) 一维卷积代替全连接
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')

            end_points = utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points
