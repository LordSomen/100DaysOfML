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

#%%

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


