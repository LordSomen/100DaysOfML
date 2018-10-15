#%%
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width
Y = (iris.target == 0).astype(np.int) # Iris Setosa?

per_clf = Perceptron()
per_clf.fit(X,Y)

y_pred = per_clf.predict([[2,0.5]])
print(y_pred) 


#%%
import tensorflow as tf

(X_train , Y_train) , (X_test,Y_test) = tf.keras.datasets.mnist.load_data()
print(X_test)
print(X_train.shape,X_test.shape)
X_train = X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test = X_test.astype(np.float32).reshape(-1,28*28)/255.0
Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)
print(X_train.shape,X_test.shape)
X_valid , X_train = X_train[:5000],X_train[5000:]
Y_valid , Y_train = Y_train[:5000],Y_train[5000:]

print(X_train)

#%%
feature_cols = [tf.feature_column.numeric_column(
    "X",shape=[28*28])]

print(feature_cols)

dnn_clf = tf.estimator.DNNClassifier(hidden_units=
[300,100],n_classes = 10,feature_columns = feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"X":X_train},y=Y_train,num_epochs=40,
    batch_size=50, shuffle = True
)
dnn_clf.train(input_fn=input_fn)

#%%
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X":X_test},y=Y_test,shuffle=False
)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)

print(eval_results)

#%%
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#%%
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.int64, shape=(None), name="Y")

#%%
def logit(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

#%%

def neuron_layer(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        W = tf.Variable(init,name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]),name="bias")
        Z = tf.matmul(X,W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

#%%

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X , n_hidden1 , name="hidden1" ,
     activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1,n_hidden2,name="hidden2",
    activation=tf.nn.relu)
    logits = neuron_layer(hidden2,n_outputs,name="outputs")

#%%

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
    Y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")


#%%

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#%%
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,Y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

#%%

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#%%
n_epochs = 40
batch_size = 50

#%%

def shuffle_batch(X,Y,batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X)//batch_size
    for batch_idx in np.array_split(rnd_idx,n_batches):
        X_batch , Y_batch = X[batch_idx],Y[batch_idx]
        yield X_batch , Y_batch

#%%

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch , Y_batch in shuffle_batch(X_train,Y_train,batch_size):
            sess.run(training_op,feed_dict={X:X_batch , Y:Y_batch})
        acc_batch = accuracy.eval(feed_dict={X:X_batch,Y:Y_batch})
        acc_val = accuracy.eval(feed_dict={X:X_valid , Y:Y_valid})
        print(epoch,"Batch accuracy:",acc_batch,"Val accuracy:",acc_val)
    
    save_path = saver.save(sess,"./my_model_final.ckpt")

#%%

with tf.Session() as sess:
    saver.restore(sess,"./my_model_final.ckpt")
    X_news_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X:X_news_scaled})
    Y_pred = np.argmax(Z,axis=1)

#%%

print("Predicted classes:",Y_pred)
print("Actual classes: ",Y_test[:20])




