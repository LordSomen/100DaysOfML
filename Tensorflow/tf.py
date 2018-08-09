#%%
import tensorflow as tf
import time

t = time.time()
x = tf.Variable(7,name="x")
y = tf.Variable(2,name="y")
f = x*x*x + y + 2
t1 = time.time() - t
print(t1)

#%%

t = time.time()
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
t1 = time.time() - t
print(t1)

#%%
t = time.time()
result = sess.run(f)
print(result)
sess.close
t1 = time.time() - t
print(t1)

#%%
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

#%%
init = tf.global_variables_initializer()
# prepare an init node
with tf.Session() as sess:
    init.run() # actually initialize all the variables
    result = f.eval()
    print(result)

#%%
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

