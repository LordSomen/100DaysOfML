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
with tf.Session() as sess:
    init.run() 
    result = f.eval()
    print(result)

#%%
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

#%%
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

#%%
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())
tf.reset_default_graph()

#%%
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

#%%
with tf.Session() as sess:
    y_val,z_val = sess.run([y,z])
    print(y_val)
    print(z_val)

#%%
import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m,n = housing.data.shape
print(m,n)

#%%
housing_data_plus_bias = np.c_[np.ones((m, 1)),
    housing.data]
print(housing_data_plus_bias.shape)
#%%
X = tf.constant(housing_data_plus_bias, dtype=tf.float32,
    name="X")
Y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,
    name='Y')
Y.shape
#%%
XT = tf.transpose(X)
XT.shape
#%%
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)),
 XT), Y)
with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)

#%%
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    print(best_theta)

#%%
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
    print(B_val_1)
    print(B_val_2)

#%%
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

#%%
def fetch_batch(epoch, batch_index, batch_size):
     # load the data from disk
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()

#%%
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0: # checkpoint every 100 epochs
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

#%%
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")