# %%
import pandas as pd
path = r"/home/soumyajit/Codes/DL/100DaysOfML/Kaggle/Titanic/train.csv"
passenger_train = pd.read_csv(path, sep=",")
passenger_train

# %%
path = r"/home/soumyajit/Codes/DL/100DaysOfML/Kaggle/Titanic/test.csv"
passenger_test = pd.read_csv(path)
passenger_test

# %%
passenger_train.describe()

# %%
passenger_train.info()

# %%
passenger_train["Pclass"].value_counts()

# %%
passenger_train["Survived"].value_counts()

# %%
passenger_train["Sex"].value_counts()

# %%
passenger_train["Embarked"].value_counts()

# %%
passenger_train["SibSp"].value_counts()

# %%
passenger_train["Parch"].value_counts()

# %%
%matplotlib inline
import matplotlib.pyplot as plt
passenger_train.hist(bins=50, figsize=(20, 15))
plt.show()

# %%
corr_matrix = passenger_train.corr()
corr_matrix["Survived"].sort_values(ascending=False)

# %%
from pandas.tools.plotting import scatter_matrix
attributes = ["Survived", "Fare", "Age", "Pclass", "Parch", "SibSp"]
scatter_matrix(passenger_train[attributes], figsize=(20, 15))

# %%
passenger_train.plot(kind="scatter", x="Survived",
                     y="Fare", alpha=0.1, figsize=(15, 12))

# %%
passenger_train_var = passenger_train
passenger_train_var["Age_Per_Fare"] = passenger_train_var["Age"]/passenger_train_var["Fare"]
passenger_train_var.plot(kind="scatter", x="Survived", y="Age_Per_Fare",
                         alpha=0.1, figsize=(15, 12))

# %%
corr_matrix = passenger_train_var.corr()
corr_matrix["Survived"].sort_values(ascending=False)

#%%
passenger_train_var = passenger_train.drop("Survived",axis = 1)
passenger_train_label = passenger_train["Survived"].copy()
passenger_train_label

#%%
passenger_train_var = passenger_train_var.drop("Name",axis = 1)
passenger_train_var = passenger_train_var.drop("Cabin",axis = 1)
passenger_train_var = passenger_train_var.drop("Ticket",axis = 1)
passenger_train_var = passenger_train_var.drop("PassengerId",axis = 1)
passenger_train_var = passenger_train_var.drop("Embarked",axis = 1)
passenger_train_var = passenger_train_var.drop("Age_Per_Fare",axis = 1)

passenger_train_var_num = passenger_train_var.drop("Sex",axis = 1)
# passenger_train_var_num = passenger_train_var_num.drop("Embarked",axis = 1)
passenger_train_var_num

#%%
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer

num_attribs = list(passenger_train_var_num)
cat_attribs = ["Sex"]
num_pipeline = Pipeline([
('selector', DataFrameSelector(num_attribs)),
('imputer', Imputer(strategy="median")),
('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
('selector', DataFrameSelector(cat_attribs)),
('label_binarizer', LabelBinarizer()),
])
full_pipeline = FeatureUnion(transformer_list=[
("num_pipeline", num_pipeline),
("cat_pipeline", cat_pipeline),
])
full_pipeline

#%%
passenger_train_var
#%%
passenger_prepared = full_pipeline.fit_transform(passenger_train_var)
passenger_prepared