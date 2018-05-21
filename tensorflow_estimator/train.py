'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: train.py
@time: 2018/5/19 下午4:46
@desc: shanghaijiaotong university
'''
import tensorflow as tf
from tensorflow.contrib import layers, slim
import os
import time
import json
from read_conf import Config
from util import list_files, elapse_time
from collections import OrderedDict
import shutil

ModeKeys = tf.estimator.ModeKeys
tf.logging.set_verbosity(tf.logging.INFO)
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
# 代入配置的方法, yaml
CONF = Config()
train_conf = CONF.train


def model(inputs, is_training, scope="linear"):
    with tf.variable_scope(scope):
        net = slim.fully_connected(inputs, 1, activation_fn=None,
                                   biases_initializer=tf.constant_initializer(0.001))
    return net


# 定义训练方法，包括优化方法和梯度预处理
def get_train_op_fn(loss, params):
    return layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer=tf.train.FtrlOptimizer(
            learning_rate=params['learning_rate'],
            l1_regularization_strength=0.5,
            l2_regularization_strength=1),
        learning_rate=params['learning_rate']
    )


# 定义评价指标
def get_eval_metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy')
    }


def model_fn(features, labels, mode, params, config):
    """Model function used in the estimator.
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    is_training = mode == ModeKeys.TRAIN
    """
    加入特征列机制
    """
    features = tf.feature_column.input_layer(features, params['feature_columns'])

    # Define model's architecture
    logits = model(features, is_training=is_training)
    predictions = tf.squeeze(tf.nn.sigmoid(logits))
    # Loss, training and eval operations are not needed during inference.
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.PREDICT:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32),
            logits=logits)

        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def parse_fn(field_delim=',', na_value='-'):
    """Parse function for csv data
        na_value: use csv defaults to fill na_value
        multivalue: bool, defaults to False
            True for csv data with multivalue features.
            eg:   f1       f2   ...
                a, b, c    1    ...
                 a, c      2    ...
                 b, c      0    ...
    Returns:
        feature dict: {feature: Tensor ... }
    """
    feature = CONF.get_feature_name()  # all features
    feature_unused = CONF.get_feature_name('unused')  # unused features
    feature_conf = CONF.read_feature_conf()  # feature conf dict

    def column_to_csv_defaults():
        """parse columns to record_defaults param in tf.decode_csv func
        Return:
            OrderedDict {'feature name': [''],...}
        """
        csv_defaults = OrderedDict()
        csv_defaults['is_trade'] = [0]  # first label default, empty if the field is must
        for f in feature:
            if f in feature_conf:  # used features
                conf = feature_conf[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':  # identity category column need int type
                        csv_defaults[f] = [0]
                    else:
                        csv_defaults[f] = ['']
                else:
                    csv_defaults[f] = [0.0]  # 0.0 for float32
            else:  # unused features
                csv_defaults[f] = ['']
        return csv_defaults

    csv_defaults = column_to_csv_defaults()

    def parser(value):
        """Parse train and eval data with label
        Args:
            value: Tensor("arg0:0", shape=(), dtype=string)
        """
        # `tf.decode_csv` return rank 0 Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>
        # na_value fill with record_defaults
        columns = tf.decode_csv(
            value, record_defaults=list(csv_defaults.values()),
            field_delim=field_delim, use_quote_delim=False, na_value=na_value)
        features = dict(zip(csv_defaults.keys(), columns))
        temp = features.copy()
        for f, tensor in temp.items():
            if f in feature_unused:
                features.pop(f)  # remove unused features
                continue
        # csv parse as string
        labels = tf.equal(temp.pop('is_trade'), "1")
        return features, labels

    return parser


def input_fn(data_file, mode, batch_size):
    dist_conf = CONF.distribution
    is_distribution = dist_conf["is_distribution"]
    cluster = dist_conf["cluster"]
    job_name = dist_conf["job_name"]
    task_index = dist_conf["task_index"]
    num_workers = 1 + len(cluster["worker"])  # must have 1 chief worker
    worker_index = task_index if job_name == "worker" else num_workers - 1
    train_conf = CONF.train
    shuffle_buffer_size = train_conf["num_examples"]
    num_parallel_calls = train_conf["num_parallel_calls"]

    tf.logging.info('Parsing input csv files: {}'.format(data_file))
    # 读取特征
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
    # Use `Dataset.map()` to build a pair of a feature dictionary
    # and a label tensor for each example.
    # Shuffle, repeat, and batch the examples.
    if is_distribution:  # allows each worker to read a unique subset.
        dataset = dataset.shard(num_workers, worker_index)

    dataset = dataset.map(
        parse_fn(),
        num_parallel_calls=num_parallel_calls)
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=123)
    # 数据集预读取
    dataset = dataset.prefetch(2 * batch_size)
    # 针对sequence feature, dynamic padding
    # batch(): each element tensor must have exactly same shape, change rank 0 to rank 1
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def train(model):
    _train_epochs = train_conf['train_epochs']
    _train_data = train_conf['train_data']
    _batch_size = train_conf['batch_size']
    for n in range(_train_epochs):
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        train_data_list = list_files(_train_data)  # dir to file list
        for f in train_data_list:
            t0 = time.time()
            tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, f))
            model.train(
                input_fn=lambda: input_fn(f, ModeKeys.TRAIN, _batch_size),
                hooks=None,
                steps=None,
                max_steps=None,
                saving_listeners=None)

            tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, f, elapse_time(t0)))
            print('-' * 80)


def train_and_eval(model):
    _train_epochs = train_conf['train_epochs']
    _train_data = train_conf['train_data']
    _eval_data = train_conf['eval_data']
    _test_data = train_conf['test_data']
    _batch_size = train_conf['batch_size']
    _epochs_per_eval = train_conf['epochs_per_eval']

    for n in range(_train_epochs):
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        train_data_list = list_files(_train_data)  # dir to file list
        for f in train_data_list:
            t0 = time.time()
            tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, f))
            model.train(
                input_fn=lambda: input_fn(f, ModeKeys.TRAIN, _batch_size),
                hooks=None,
                steps=None,
                max_steps=None,
                saving_listeners=None)

            tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, f, elapse_time(t0)))
            print('-' * 80)

            tf.logging.info('<EPOCH {}>: Start evaluating {}'.format(n + 1, _eval_data))
            t0 = time.time()

            results = model.evaluate(
                input_fn=lambda: input_fn(f, ModeKeys.EVAL, _batch_size),
                steps=None,  # Number of steps for which to evaluate model.
                hooks=None,
                checkpoint_path=None,  # latest checkpoint in model_dir is used.
                name=None)

            tf.logging.info(
                '<EPOCH {}>: Finish evaluation {}, take {} mins'.format(n + 1, _eval_data, elapse_time(t0)))
            print('-' * 80)
            # Display evaluation metrics
            for key in sorted(results):
                print('{}: {}'.format(key, results[key]))

        # every epochs_per_eval test the model (use larger test dataset)
        if (n + 1) % _epochs_per_eval == 0:
            tf.logging.info('<EPOCH {}>: Start testing {}'.format(n + 1, _test_data))
            results = model.evaluate(
                input_fn=lambda: input_fn(f, ModeKeys.EVAL, _batch_size),
                steps=None,  # Number of steps for which to evaluate model.
                hooks=None,
                checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                name=None)

            tf.logging.info(
                '<EPOCH {}>: Finish testing {}, take {} mins'.format(n + 1, _test_data, elapse_time(t0)))
            print('-' * 80)
            # Display evaluation metrics
            for key in sorted(results):
                print('{}: {}'.format(key, results[key]))


def build_distribution():
    """Build distribution configuration variable TF_CONFIG in tf.estimator API"""
    TF_CONFIG = CONF.distribution
    if TF_CONFIG["is_distribution"]:
        cluster_spec = TF_CONFIG["cluster"]
        job_name = TF_CONFIG["job_name"]
        task_index = TF_CONFIG["task_index"]
        os.environ['TF_CONFIG'] = json.dumps(
            {'cluster': cluster_spec,
             'task': {'type': job_name, 'index': task_index}})
        run_config = tf.estimator.RunConfig()
        if job_name in ["ps", "chief", "worker"]:
            assert run_config.master == 'grpc://' + cluster_spec[job_name][task_index]  # grpc://10.120.180.212
            assert run_config.task_type == job_name
            assert run_config.task_id == task_index
            assert run_config.num_ps_replicas == len(cluster_spec["ps"])
            assert run_config.num_worker_replicas == len(cluster_spec["worker"]) + len(cluster_spec["chief"])
            assert run_config.is_chief == (job_name == "chief")
        elif job_name == "evaluator":
            assert run_config.master == ''
            assert run_config.evaluator_master == ''
            assert run_config.task_id == 0
            assert run_config.num_ps_replicas == 0
            assert run_config.num_worker_replicas == 0
            assert run_config.cluster_spec == {}
            assert run_config.task_type == 'evaluator'
            assert not run_config.is_chief


def build_estimator(model_dir, model_type):
    wide_columns, deep_columns = build_model_columns()
    build_distribution()
    config = tf.ConfigProto(device_count={"GPU": 0},  # limit to GPU usage
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1,
                            log_device_placement=True)
    run_config = tf.estimator.RunConfig(**CONF.runconfig).replace(session_config=config)
    params = {
        'feature_columns': wide_columns,
        'learning_rate': 0.001
    }

    return tf.estimator.Estimator(
        model_dir=model_dir,
        model_fn=model_fn,
        params=params,
        config=run_config
    )


import numpy as np

# wide columns
categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
crossed_column = tf.feature_column.crossed_column
bucketized_column = tf.feature_column.bucketized_column
# deep columns
embedding_column = tf.feature_column.embedding_column
indicator_column = tf.feature_column.indicator_column
numeric_column = tf.feature_column.numeric_column


def build_model_columns():
    """
    Build wide and deep feature columns from custom feature conf using tf.feature_column API
    wide_columns: category features + cross_features + [discretized continuous features]
    deep_columns: continuous features + category features(onehot or embedding for sparse features) + [cross_features(embedding)]

    Return:
        _CategoricalColumn and __DenseColumn instance in tf.feature_column API
    """

    def embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim ** 0.25))))

    def normalizer_fn_builder(scaler, normalization_params):
        """normalizer_fn builder"""
        if scaler == 'min_max':
            return lambda x: (x - normalization_params[0]) / (x - normalization_params[1])
        elif scaler == 'standard':
            return lambda x: (x - normalization_params[0]) / normalization_params[1]
        else:
            return lambda x: tf.log(x)

    # 读取conf，并设置特殊的针对特征的预处理方法（针对columns）
    feature_conf_dic = CONF.read_feature_conf()
    cross_feature_list = CONF.read_cross_feature_conf()
    tf.logging.info('Total used feature class: {}'.format(len(feature_conf_dic)))
    tf.logging.info('Total used cross feature class: {}'.format(len(cross_feature_list)))
    wide_columns = []
    deep_columns = []
    wide_dim = 0
    deep_dim = 0
    for feature, conf in feature_conf_dic.items():
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
        if f_type == 'category':
            if f_tran == 'hash_bucket':
                hash_bucket_size = f_param
                embed_dim = embedding_dim(hash_bucket_size)
                col = categorical_column_with_hash_bucket(feature,
                                                          hash_bucket_size=hash_bucket_size,
                                                          dtype=tf.string)
                wide_columns.append(indicator_column(col))
                # wide_columns.append(col)
                deep_columns.append(embedding_column(col,
                                                     dimension=embed_dim,
                                                     combiner='mean',
                                                     initializer=None,
                                                     ckpt_to_load_from=None,
                                                     tensor_name_in_ckpt=None,
                                                     max_norm=None,
                                                     trainable=True))
                wide_dim += hash_bucket_size
                deep_dim += embed_dim
            elif f_tran == 'vocab':
                col = categorical_column_with_vocabulary_list(feature,
                                                              vocabulary_list=list(map(str, f_param)),
                                                              dtype=None,
                                                              default_value=-1,
                                                              num_oov_buckets=0)  # len(vocab)+num_oov_bucket
                wide_columns.append(col)
                deep_columns.append(indicator_column(col))
                wide_dim += len(f_param)
                deep_dim += len(f_param)

            elif f_tran == 'identity':
                num_buckets = f_param
                col = categorical_column_with_identity(feature,
                                                       num_buckets=num_buckets,
                                                       default_value=0)  # Values outside range will result in default_value if specified, otherwise it will fail.
                wide_columns.append(col)
                deep_columns.append(indicator_column(col))
                wide_dim += num_buckets
                deep_dim += num_buckets

        # 连续值
        else:
            normalizaton, boundaries = f_param["normalization"], f_param["boundaries"]
            if f_tran is None:
                normalizer_fn = None
            else:
                normalizer_fn = normalizer_fn_builder(f_tran, tuple(normalizaton))
            col = numeric_column(feature,
                                 shape=(1,),
                                 default_value=0,  # default None will fail if an example does not contain this column.
                                 dtype=tf.float32,
                                 normalizer_fn=normalizer_fn)

            if boundaries:  # whether include continuous features in wide part
                wide_columns.append(bucketized_column(col, boundaries=boundaries))
                wide_dim += (len(boundaries) + 1)

            deep_columns.append(col)
            deep_dim += 1

    for cross_features, hash_bucket_size, is_deep in cross_feature_list:
        cf_list = []
        for f in cross_features:
            f_type = feature_conf_dic[f]["type"]
            f_tran = feature_conf_dic[f]["transform"]
            f_param = feature_conf_dic[f]["parameter"]
            if f_type == 'continuous':
                cf_list.append(bucketized_column(numeric_column(f, default_value=0), boundaries=f_param['boundaries']))
            else:
                if f_tran == 'identity':
                    # If an input feature is of numeric type, you can use categorical_column_with_identity
                    cf_list.append(categorical_column_with_identity(f, num_buckets=f_param,
                                                                    default_value=0))
                else:
                    cf_list.append(f)  # category col put the name in crossed_column
        col = crossed_column(cf_list, hash_bucket_size)
        wide_columns.append(indicator_column(col))
        # wide_columns.append(col)
        wide_dim += hash_bucket_size
        if is_deep:
            deep_columns.append(embedding_column(col, dimension=embedding_dim(hash_bucket_size)))
            deep_dim += embedding_dim(hash_bucket_size)

    # add columns logging info
    tf.logging.info('Build total {} wide columns'.format(len(wide_columns)))
    for col in wide_columns:
        tf.logging.debug('Wide columns: {}'.format(col))
    tf.logging.info('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        tf.logging.debug('Deep columns: {}'.format(col))
    tf.logging.info('Wide input dimension is: {}'.format(wide_dim))
    tf.logging.info('Deep input dimension is: {}'.format(deep_dim))
    return wide_columns, deep_columns

def run():
    print("Using TensorFlow Version %s" % tf.__version__)
    assert "1.4" <= tf.__version__, "Need TensorFlow r1.4 or Later."

    print("\nUsing Train Config:")
    for k, v in CONF.train.items():
        print('{}: {}'.format(k, v))

    print("\nUsing Model Config:")
    for k, v in CONF.model.items():
        print('{}: {}'.format(k, v))

    model_dir = os.path.join(train_conf['model_dir'], train_conf['model_type'])

    if not train_conf['keep_train']:
        # Clean up the model directory if not keep training
        shutil.rmtree(model_dir, ignore_errors=True)
        print('Remove model directory: {}'.format(model_dir))

    estimator = build_estimator(model_dir, train_conf['model_type'])
    tf.logging.info('Build estimator: {}'.format(estimator))
    # distributed can not including eval
    if CONF.distribution["is_distribution"]:
        print("Using PID: {}".format(os.getpid()))
        cluster = CONF.distribution["cluster"]
        job_name = CONF.distribution["job_name"]
        task_index = CONF.distribution["task_index"]
        print("Using Distributed TensorFlow. Local host: {} Job_name: {} Task_index: {}"
              .format(cluster[job_name][task_index], job_name, task_index))
        cluster = tf.train.ClusterSpec(CONF.distribution["cluster"])
        server = tf.train.Server(cluster,
                                 job_name=job_name,
                                 task_index=task_index)
        if job_name == 'ps':
            server.join()
        else:
            train(estimator)
    else:
        train_and_eval(estimator)


if __name__ == '__main__':
    run()
