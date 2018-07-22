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