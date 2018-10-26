#%%
import tensorflow as tf
import numpy as np

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
