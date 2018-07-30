#%%
''' retrieving mnist data '''

from sklearn.datasets import fetch_mldata
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
print(X_val)

#%%
