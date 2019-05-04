import tensorflow as tf


class CNNConfig(object):
    """CNN参数"""
    # input_data.shape = [seq_length, input_dim]
    input_dim = 6
    seq_length = 100

    num_filters = [32, 64, 128]   # 卷积核数目
    hidden_dim = 128              # 全连接层神经元
    num_classes = 10              # 类别
    kernel_size = 5               # 卷积核大小
    pool_size = 3                 # 池化大小

    dropout_keep_prob = 0.5
    learning_rate = 1e-3

    batch_size = 128
    num_epochs = 20
    print_per_batch = 10          # 每多少轮输出一次结果


class CNN(object):
    """CNN model"""
    def __init__(self, config):
        self.config = config
        # input_x, input_y, keep_prob
        self.input_x = tf.placeholder(tf.float32,
                                      [None, self.config.seq_length, self.config.input_dim],
                                      name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv1d(self.input_x, self.config.num_filters[0], self.config.kernel_size,
                                     padding='SAME', strides=1, activation=tf.nn.relu, name='conv1')

        with tf.name_scope('pooling2'):
            pool1 = tf.layers.max_pooling1d(conv1, self.config.pool_size, strides=2, name='pooling1')

        with tf.name_scope('conv2'):
            conv2 = tf.layers.conv1d(pool1, self.config.num_filters[1], self.config.kernel_size,
                                     padding='SAME', strides=1, activation=tf.nn.relu, name='conv2')

        with tf.name_scope('pooling2'):
            pool2 = tf.layers.max_pooling1d(conv2, self.config.pool_size, strides=2, name='pooling2')
        with tf.name_scope('conv3'):
            conv3 = tf.layers.conv1d(pool2, self.config.num_filters[2], self.config.kernel_size,
                                     padding='SAME', strides=1, activation=tf.nn.relu, name='conv3')
        with tf.name_scope('pooling3'):
            pool3 = tf.layers.max_pooling1d(conv3, self.config.pool_size, strides=2, name='pooling3')

        with tf.name_scope('score'):
            temp = tf.reshape(pool3, (-1, pool3.shape[1] * pool3.shape[2]))
            fc = tf.layers.dense(temp, self.config.hidden_dim, activation=tf.nn.relu, name='fc')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.arg_max(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('optimize'):
            # 损失函数
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.arg_max(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
