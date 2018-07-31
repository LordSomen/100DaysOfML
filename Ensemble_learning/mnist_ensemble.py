#%%
''' retrieving mnist data '''

from sklearn.datasets import fetch_mldata
import numpy as np
mnist = fetch_mldata('MNIST original')
print(mnist)

#%%
X, Y = mnist["data"], mnist["target"]
print(X)
print(Y)
print(X.shape)
print(Y.shape)

#%%
X_train,Y_train,X_val,Y_val,X_test,Y_test = X[:40000],Y[:40000],X[40000:50000],Y[40000:50000],X[50000:],Y[50000:]
shuffle_index = np.random.permutation(40000)
X_train, Y_train = X_train[shuffle_index], Y_train[shuffle_index]
X_train_array = np.array(X_train)
Y_train_array = np.array(Y_train)
X_val_array = np.array(X_val)
Y_val_array = np.array(Y_val)
X_test_array = np.array(X_test)
Y_test_array = np.array(Y_test)

print(X_val)

#%%
''' training of RandomForestClassifier'''
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=1000,n_jobs=100,
random_state=42)
rnd_clf.fit(X_train_array,Y_train_array)
rnd_pred = rnd_clf.predict(X_val_array)
rnd_eff = accuracy_score(Y_val_array,rnd_pred)
print(rnd_eff)