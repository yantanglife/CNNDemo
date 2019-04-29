import os
import sys
import tensorflow as tf
from cnn_demo import CNNConfig, CNN
from data_process import get_batch, get_train_valid_test_data


save_dir = './checkpoints'
save_path = os.path.join(save_dir, 'best_validation')

x_train, x_val, x_test, y_train, y_val, y_test = get_train_valid_test_data()


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = get_batch(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    flag = False

    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)

        batch_train = get_batch(x_train, y_train, config.batch_size)
        # print(batch_train)
        for x_batch, y_batch in batch_train:

            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            # print(y_batch)
            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(sess, x_val, y_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%},  {5}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))

            sess.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test():
    print()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=save_path)

    print('Testing...')
    loss_test, acc_test = evaluate(sess, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3] or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python *.py [train / test]""")
    config = CNNConfig()
    model = CNN(config)
    if sys.argv[1] == 'train':
        train()
    if (len(sys.argv) == 3 and sys.argv[2] == 'test') or sys.argv[1] == 'test':
        test()
