#%%
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

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit(X)
print(X_2d)

#%%
'''choosing n_components effieciently'''
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(d)

#%%
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)

#%%
pca = PCA(n_components = 154)
X_mnist_reduced = pca.fit_transform(X)
X_mnist_recovered = pca.inverse_transform(X_mnist_reduced)

#%%
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X_batch)
X_mnist_reduced = inc_pca.transform(X)

#%%
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X)