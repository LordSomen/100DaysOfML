#%%
'''
Experimenting on MNIST dataset.
First training it on the RandomTreeClassifier, and Noting the time.
Then reducing the dimensionality of the dataset we are training on the same
classifier, note the time to see which one take less time.
'''
from sklearn.datasets import fetch_mldata
import numpy as np

mnist = fetch_mldata("MNIST original")
print(mnist)

#%%
X,Y = mnist['data'],mnist['target']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=10000,
random_state=42)
print(X_train.shape)
print(Y_train.shape)

#%%
from sklearn.ensemble import RandomForestClassifier
import time
rnd_clf = RandomForestClassifier(n_estimators=1000,n_jobs=10,
random_state=42) 
t1 = time.time()
rnd_clf.fit(X_train,Y_train)
print ( "training time:", round(time.time()-t1, 3), "s") 
print("process time:",time.process_time)
print(rnd_clf.score(X_test,Y_test))
# the time would be round to 3 decimal in seconds

##%%
'''
reducing the dimensionality of the 
mnist dataset
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X_reduced,Y,
test_size=10000,random_state=42)
print(X_train.shape)
print(Y_train.shape)
t2 = time.time()
rnd_clf.fit(X_train,Y_train)
print ( "training time:", round(time.time()-t2, 3), "s") 
print("process time:",time.process_time)
print(rnd_clf.score(X_test,Y_test))
