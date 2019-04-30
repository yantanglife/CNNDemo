import tensorflow as tf


class RNNConfig(object):
    input_dim = 6
    seq_length = 100
    num_classes = 10

    num_layers = 2
    hidden_dim = 128
    rnn = 'lstm'

    dropout_keep_prob = 0.9
    learning_rate = 1e-3

    batch_size = 64
    num_epochs = 20

    print_per_batch = 10


class RNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.float32,
                                      [None, self.config.seq_length, self.config.input_dim],
                                      name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.rnn()

    def rnn(self):
        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim)

        def gru_cell():
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)

        def dropout():
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.name_scope('rnn'):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.input_x, dtype=tf.float32)
            # 取最后一个时序输出作为结果
            last = _outputs[:, -1, :]
        print(_outputs, last)
        with tf.name_scope('score'):
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            print(fc)
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            print(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
            print(self.y_pred_cls)
        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.arg_max(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
