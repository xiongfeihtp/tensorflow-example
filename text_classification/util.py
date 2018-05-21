import tensorflow as tf
def get_record_parser(config):
    def parse(example):
        para_limit = config.sequence_len
        features = tf.parse_single_example(example,
                                           features={
                                               "text": tf.FixedLenFeature([], tf.string),
                                               "label": tf.FixedLenFeature([], tf.string)
                                           })
        # 因为设置了固定的对齐长度，所以可以直接进行reshape恢复成正常的tensor
        text = tf.reshape(tf.decode_raw(
            features["text"], tf.int32), [para_limit])
        label = tf.reshape(tf.decode_raw(
            features["label"], tf.int32),[1])
        return text, label
    return parse
#训练数据迭代器
def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    dataset = dataset.batch(config.batch_size)
    return dataset


#测试数据迭代器
def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset
