import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()

# --------------- tf.layters.conv1d -------------------
inputs=tf.ones((2,10,2))  # [batch, n_sqs, embedsize]
num_filters=3
kernel_size =3
conv2 = tf.layers.conv1d(inputs, num_filters, kernel_size,strides=1, padding='same',name='conv2')  # shape = (batchsize, round(n_sqs/strides),num_filters)
tf.global_variables_initializer().run()
out = sess.run(conv2)
print(out)
pool1 = tf.layers.max_pooling1d(conv2, 3, strides=1, name='pooling1')
out2 = sess.run(pool1)
print(out2)
'''
sess = tf.InteractiveSession()

# --------------- tf.nn.conv1d  -------------------
inputs=tf.ones((2, 10, 3))  # [batch, n_sqs, embedsize]
w=tf.constant(1, tf.float32, (5,3,5))  # [w_high, embedsize, n_filers]
conv1 = tf.nn.conv1d(inputs,w,stride=1 ,padding='SAME')  # conv1=[batch, round(n_sqs/stride), n_filers],stride是步长。

tf.global_variables_initializer().run()
out = sess.run(conv1)
print(out)
'''