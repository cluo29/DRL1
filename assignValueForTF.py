
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
loss = tf.reduce_sum(tf.abs(tf.subtract(x, y_)))#Function chosen arbitrarily
input_x=np.random.randn(100, 2)#Random generation of variable x
input_y=np.random.randn(100, 2)#Random generation of variable y

with tf.Session() as sess:
    print(sess.run(loss, feed_dict={x: input_x, y_: input_y}))