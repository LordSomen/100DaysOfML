#%%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris['data'][:,:2]
Y = iris['target']

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,Y)

#%%
from sklearn.tree import export_graphviz
export_graphviz(
tree_clf,
out_file="Decision_Trees/iris_tree.dot",
feature_names=iris['feature_names'][2:],
class_names=iris['target_names'],
rounded=True,
filled=True
)

#%%
tree_clf.predict_proba([7,1.5])

#%%
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, Y)

