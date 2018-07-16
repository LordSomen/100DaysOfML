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

#%%
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X,Y = make_moons()
print(X)
print(Y)

#%%
'''
for non linear distinction datasets 
'''
polynomial_svm_clf = Pipeline((
    ("poly_features",PolynomialFeatures()),
    ('scaler',StandardScaler()),
    ('linear_svc',LinearSVC(C=10,loss="hinge")),
))

polynomial_svm_clf.fit(X,Y)

#%%
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))
poly_kernel_svm_clf.fit(X, Y)

#%%
rbf_kernel_svm_clf = Pipeline((
    ("scaler",StandardScaler()),
    ("svm_clf",SVC(kernel='rbf',gamma=5,C=0.001))
))
rbf_kernel_svm_clf.fit(X,Y)

#%%
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X,Y)

#%%
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly",degree=2,epsilon=0.01,C=100)
svm_poly_reg.fit(X,Y)


