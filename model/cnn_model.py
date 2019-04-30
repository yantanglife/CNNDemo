import tensorflow as tf


class CNNConfig(object):
    # input_data.shape = [seq_length, input_dim]
    input_dim = 6
    # input_chanel = 2
    seq_length = 100

    num_filters = [32, 64, 128]
    hidden_dim = 128
    num_classes = 10
    kernel_size = 5

    print_per_batch = 10

    dropout_keep_prob = 0.5
    learning_rate = 1e-3

    batch_size = 128
    num_epochs = 20


class CNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.float32,
                                      [None, self.config.seq_length, self.config.input_dim],
                                      name='input_x')
        self.   input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv1d(self.input_x, self.config.num_filters[0], self.config.kernel_size,
                                     padding='SAME', strides=1, activation=tf.nn.relu, name='conv1')

        with tf.name_scope('pooling2'):
            pool1 = tf.layers.max_pooling1d(conv1, 3, strides=2, name='pooling1')

        with tf.name_scope('conv2'):
            conv2 = tf.layers.conv1d(pool1, self.config.num_filters[1], self.config.kernel_size,
                                     padding='SAME', strides=1, activation=tf.nn.relu, name='conv2')

        with tf.name_scope('pooling2'):
            pool2 = tf.layers.max_pooling1d(conv2, 3, strides=2, name='pooling2')
        with tf.name_scope('conv3'):
            conv3 = tf.layers.conv1d(pool2, self.config.num_filters[2], self.config.kernel_size,
                                     padding='SAME', strides=1, activation=tf.nn.relu, name='conv3')
        with tf.name_scope('pooling3'):
            pool3 = tf.layers.max_pooling1d(conv3, 3, strides=2, name='pooling3')

        with tf.name_scope('score'):
            temp = tf.reshape(pool3, (-1, pool3.shape[1] * pool3.shape[2]))
            fc = tf.layers.dense(temp, self.config.hidden_dim, activation=tf.nn.relu, name='fc')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.arg_max(tf.nn.softmax(self.logits), 1)
        # print(conv1, pool1, conv2, pool2, fc, self.logits, self.y_pred_cls)
        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.arg_max(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
