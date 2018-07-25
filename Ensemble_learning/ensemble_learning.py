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