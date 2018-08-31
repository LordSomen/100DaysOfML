#%%
import tensorflow as tf 
x = tf.Variable(tf.random_uniform(shape=(),minval=0.0,maxval=1.0))
x_new_val = tf.placeholder(shape=(),dtype=tf.float32)
x_assign = tf.assign(x, x_new_val)
with tf.Session():
    x.initializer.run() 
    print(x.eval()) 
    x_assign.eval(feed_dict={x_new_val: 5.0})
    print(x.eval()) 