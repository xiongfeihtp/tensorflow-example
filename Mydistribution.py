'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: Mydistribution.py
@time: 2018/4/14 下午8:42
@desc: shanghaijiaotong university
'''
import tensorflow as tf
import tempfile
import time

batch_size = 128
max_steps = 1000000
num_gpus = 4

flags = tf.flags
tf.flags.DEFINE_string(name='job_name', default="", help="job name")
tf.flags.DEFINE_integer(name='task_index', default=0, help='task id for special job')
tf.flags.DEFINE_string(name='ps_hosts', default="", help="ps hosts")
tf.flags.DEFINE_string(name='worker_hosts', default="", help="worker hosts")

config = flags.FLAGS


def main():
    # data_generator
    if config.job_name is None or config.job_name == "":
        raise ValueError("Must specify an explicit job_name")
    if config.task_index is None or config.task_index == "":
        raise ValueError("Must specify an explicit task_index")
    print("job name = {}".format(config.job_name))
    print("task name = {}".format(config.task_index))

    # server和worker的地址
    ps_spec = config.ps_hosts.split(",")
    worker_spec = config.worker_hosts.split(",")

    num_workers = len(worker_spec)
    #创建集群context，并且设置server，并连接集群
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    server = tf.train.Server(cluster, job_name=config.job_name, task_index=config.task_index)
    """
    前面都是相同的部分，后面开始不同
    """
    # 如果是parameter server，则阻塞，并等待
    if config.job_name == "ps":
        server.join()

    elif config.job_name == "worker":
        # workers部分，判断是否为主节点（进行全局参数初始化，并启动队列）
        is_chief = (config.task_index == 0)
        # 设置task对应的计算资源群和存储参数资源群
        worker_device = "/job:worker/task:{}/gpu:0".format(config.task_index)
        # 通过device设置资源
        with tf.device(
                tf.train.replica_device_setter(
                    worker_device=worker_device,  # 计算资源
                    ps_device="/job:ps/cpu:0",  # 存储模型参数资源，参数服务器，如果不设置则自己循环分发
                    cluster=cluster
                )):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            """
            model definition
            
            """
            loss = None
            opt = tf.train.AdamOptimizer(config.learning_rate)
            """
            设置优化方法
            """
            # 同步模型
            # 全局batch更新累积到一定数量进行梯度跟新
            if config.sync_replicas:
                if config.replicas_to_aggregate is None:
                    replicas_to_aggregate = num_workers
                else:
                    replicas_to_aggregate = config.replicas_to_aggregate
                # 封装opt
                opt = tf.train.SyncReplicasOptimizer(
                    opt,
                    replicas_to_aggregate=replicas_to_aggregate,
                    total_num_replicas=num_workers,
                    name="sync_replics"
                )
            # 异步模型，不需要封装
            train_step = opt.minimize(loss, global_step=global_step)
        """
        初始化和配置
        """
        # 如果是同步训练模式，并且为主节点，创建队列执行器和全局初始化器
        if config.sync_replicas and is_chief:
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op()
        # 创建临时目录
        train_dir = tempfile.mkdtemp()
        init_op = tf.global_variables_initializer()
        # 分布式模式下，控制参数初始化和model save, model load, log
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            recovery_wait_secs=1,
            global_step=global_step
        )
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,  # 是否记录日志
            # 这个参数是硬件过滤器，如果被设置的话，会话会忽略掉所有不匹配过滤器的硬件。
            device_filters=["/job:ps", "/job:worker/task:{}".format(config.task_index)]
        )
        if is_chief:
            print("Worker {}: Initializing session...".format(config.task_index))
        else:
            print("Worker {}: Waiting for session to be initialized...".format(config.task_index))

        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        print("Worker {}: Session initialization complete".format(config.task_index))
        if config.sync_replicas and is_chief:
            print("Staring chief queue runner and running init_tokens_op")
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)
        """
        训练
        """
        time_begin = time.time()
        print("Training begin {}".format(time_begin))
        local_step = 0
        while True:
            """
            data_generator
            """
            """
            train sess run and get global_step
            """
            global_step = None
            local_step += 1
            now = time.time()
            """
            打印训练信息
            """
            if global_step >= config.train_steps:
                break

"""
运行，分别在三个机器上运行这个程序，并且分配不同的job_name和task_index

"""

"""
思考：数据并行模式，数据的存放方式，和epoch shuffle的操作如何实现
"""
