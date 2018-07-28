#%%
from sklearn.datasets import make_moons

X,Y = make_moons(n_samples=10000)
print(X)
print(Y)

#%%
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test= train_test_split(
    X,Y,test_size=0.2,random_state=42)
print(X_train)
print(Y_train)

#%%
''' voting classifier'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

log_clf = LogisticRegression()
rand_clf = RandomForestClassifier()
svc_clf = SVC()

vote_clf = VotingClassifier(estimators=[
    ('lc',log_clf),('rc',rand_clf),('sc',svc_clf)],voting='hard')

vote_clf.fit(X_train,Y_train)

#%%
from sklearn.metrics import accuracy_score
for clf in (log_clf, rand_clf, svc_clf, vote_clf):   
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(Y_test, Y_pred))

#%%
''' bagging and pasting'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(),
            n_estimators=500,
            max_samples=100,bootstrap=False,n_jobs=1)

bag_clf.fit(X_train,Y_train)

#%%
prediction = bag_clf.predict(X_test)
print(accuracy_score(Y_test,prediction))

#%%
'''out of bag evaluation'''
bag_clf = BaggingClassifier(DecisionTreeClassifier(),
            n_estimators=500,
            max_samples=100,bootstrap=True,n_jobs=-1,
            oob_score=True)
bag_clf.fit(X_train,Y_train)
print(bag_clf.oob_score_)

#%%
'''Random Forest'''
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,
            n_jobs=-1)
rnd_clf.fit(X_train,Y_train)
Y_pred = rnd_clf.predict(X_test)
print(accuracy_score(Y_test,Y_pred))

#%%
'''Equivalent BaggingClassifier of the above randomForestCLassifier'''
bag_clf = BaggingClassifier(
DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
)

#%%
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

#%%
'''Adative boostng'''
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
DecisionTreeClassifier(max_depth=1), n_estimators=200,
algorithm="SAMME.R", learning_rate=0.5
)
ada_clf.fit(X_train, Y_train)

#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris['data'][:,:2]
Y = iris['target']

tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, Y)

#%%
Y2 = Y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X,Y2)

#%%
Y3 = Y - tree_reg2.predict(X_train)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, Y3)

#%%
Y_pred = sum(tree.predict(X) 
for tree in (tree_reg1, tree_reg2, tree_reg3))
print(Y_pred)

#%%
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, 
learning_rate=1.0)
gbrt.fit(X, Y)

#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, Y_train)
errors = [mean_squared_error(Y_val, y_pred)
for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)
gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, Y_train)

#%%
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, Y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(Y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
    if error_going_up == 5:
        break # early stopping