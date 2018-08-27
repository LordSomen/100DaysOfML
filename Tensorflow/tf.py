#%%
import tensorflow as tf
import time
import numpy as np

t = time.time()
x = tf.Variable(7,name="x")
y = tf.Variable(2,name="y")
f = x*x*x + y + 2
t1 = time.time() - t
print(t1)

#%%
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    print("Graph is Resetted")

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
reset_graph()
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
reset_graph()
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
'''scaling the data for gradient descent'''
reset_graph()
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_with_bias = np.c_[np.ones((m,1)),scaled_housing_data]
print(scaled_housing_data_with_bias)

#%%
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_with_bias, dtype=tf.float32, name="X")
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
reset_graph()
n_epochs = 1000
learning_rate = 0.01
#%%
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
#%%
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0,
 seed=42), name="theta")
Y_pred = tf.matmul(X, theta, name="predictions")
error = Y_pred - Y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=
learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
print("Graph is initialized")

#%%
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

#%%
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  
    indices = np.random.randint(m, size=batch_size)  
    X_batch = scaled_housing_data_with_bias[indices] 
    Y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, Y_batch

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, Y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
    best_theta = theta.eval()


print(best_theta)

#%%
'''saving and restoring of model'''
reset_graph()
n_epochs = 1000
learning_rate = 0.1
X = tf.constant(scaled_housing_data_with_bias,dtype=tf.float32,
name='X')
Y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,
name='Y')
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed=42),
name="theta")
Y_pred = tf.matmul(X,theta,name="predictions")
error = Y_pred - Y
mse = tf.reduce_mean(tf.square(error),name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=
learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if((epoch % 100) == 0):
            print("Epoch:",epoch,"Mse:",mse.eval())
            save_path = saver.save(sess,"/tmp/my_model.ckpt")
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess,"/tmp/my_model_final.ckpt")

#%%
print(best_theta)

#%%
with tf.Session() as sess:
    saver.restore(sess,"/tmp/my_model_final.ckpt")
    best_theta_restore = theta.eval()
print(best_theta_restore)

#%%
np.allclose(best_theta, best_theta_restore)

#%%
# if want to retrive the saver variable 
# with a different name
saver = tf.train.Saver({"weights": theta})

#%%
reset_graph()
# this loads the graph structure
saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  
theta = tf.get_default_graph().get_tensor_by_name("theta:0") 
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")  
    best_theta_restored = theta.eval() 
print(best_theta_restored)
#%%
np.allclose(best_theta, best_theta_restored)

#%%
''' implementing tensorboard'''
reset_graph()

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf.logs"
logdir = "{}/run-{}/".format(root_logdir, now)

#%%
n_epochs = 1000
learning_rate = 0.01
X = tf.placeholder(tf.float32,shape=(None,n+1),name="X")
Y = tf.placeholder(tf.float32,shape=(None,1),name="Y")
theta = tf.Variable(tf.random_uniform([n+1,1],
-1.0,1.0,seed=42),name="theta")
Y_pred = tf.matmul(X,theta,name="predictions")
error = Y_pred - Y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=
learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

#%%
mse_summary = tf.summary.scalar("MSE",mse)
file_writer = tf.summary.FileWriter(logdir,
tf.get_default_graph())

#%%
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

#%%
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch,Y_batch = fetch_batch(epoch,batch_index
            ,batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict=
                {X: X_batch, Y: Y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, 
            Y: Y_batch})

    best_theta = theta.eval()

#%%
file_writer.close()
print(best_theta)
