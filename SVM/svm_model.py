#%%
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:,(2,3)]
Y = (iris["target"]==2).astype(np.float64)
Y

#%%
svm_clf = Pipeline((
    ("scaler",StandardScaler()),
    ("linear_svc",LinearSVC(C=1.0,loss='hinge')),
    ))
svm_clf
#%%
X_scaled  = StandardScaler(X)
svm_clf.fit(X,Y)
svm_clf.predict([[5.5,1.7]])