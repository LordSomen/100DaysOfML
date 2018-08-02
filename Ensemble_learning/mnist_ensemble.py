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

from sklearn.model_selection import train_test_split
X_train_val,X_test,Y_train_val,Y_test = train_test_split(X,Y,
test_size=10000,random_state=42)
X_train,X_val,Y_train,Y_val = train_test_split(X_train_val,Y_train_val,
test_size=10000,random_state=42)
print(X_val)

#%%
''' training of RandomForestClassifier'''
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=1000,n_jobs=100,
random_state=42)
rnd_clf.fit(X_train,Y_train)
rnd_pred = rnd_clf.predict(X_test)
rnd_eff = accuracy_score(Y_test,rnd_pred)
print(rnd_eff)
print(rnd_clf.score(X_val,Y_val))

##%%

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
svm_clf = Pipeline((
    ("scaler",StandardScaler()),
    ("svm_clf",SVC(kernel='rbf',gamma=5,C=0.001))
))
svm_clf.fit(X_train,Y_train)
svm_pred = svm_clf.predict(X_test)
svm_eff = accuracy_score(Y_test,svm_pred)
print(svm_eff)
print(svm_clf.score(X_val,Y_val))

##%%
from sklearn.ensemble import ExtraTreesClassifier

extra_trees_clf = ExtraTreesClassifier(random_state=42)
extra_trees_clf.fit(X_train,Y_train)
extra_trees_pred = extra_trees_clf.predict(X_test)
extra_trees_eff = accuracy_score(Y_test,extra_trees_pred)
print(extra_trees_eff)
print(extra_trees_clf.score(X_val,Y_val))

##%%
from sklearn.ensemble import VotingClassifier

vote_clf = VotingClassifier(estimators=[
    ('svc',svm_clf),('rnd',rnd_clf),('extra_cls',extra_trees_clf)]
    ,voting='hard')

vote_clf.fit(X_train,Y_train)
vote_clf_predict = extra_trees_clf.predict(X_test)
vote_clf_eff = accuracy_score(Y_test,vote_clf_predict)
print(vote_clf_predict)
print(vote_clf.score(X_val,Y_val))
