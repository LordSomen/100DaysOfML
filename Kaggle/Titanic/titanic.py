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

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, passenger_prepared, passenger_train_label, cv=3,
method="predict_proba")
y_probas_forest

#%%
from sklearn.metrics import roc_auc_score
y_scores_forest = y_probas_forest[:, 1]
roc_auc_score(passenger_train_label, y_scores_forest)
#%%
forest_clf.fit(passenger_prepared,passenger_train_label)
forest_clf.predict(passenger_prepared[800])

#%%
passenger_test_var = passenger_test
passenger_test_var = passenger_test_var.drop("Name",axis = 1)
passenger_test_var = passenger_test_var.drop("Cabin",axis = 1)
passenger_test_var = passenger_test_var.drop("Ticket",axis = 1)
passenger_test_var = passenger_test_var.drop("PassengerId",axis = 1)
passenger_test_var = passenger_test_var.drop("Embarked",axis = 1)
passenger_test_var

#%%
passenger_prepared_test = full_pipeline.fit_transform(passenger_test_var)

final_result = forest_clf.predict(passenger_prepared_test)
final_result 

#%%
final_result_list = final_result.tolist()
final_result_list

#%%
passenger_id = passenger_test["PassengerId"].copy()
passenger_id_list = passenger_id.tolist()
passenger_id_list

#%%
final_result_combined_list = []
for i,j in zip(passenger_id_list,final_result_list):
    final_result_combined_list.append([i,j])
final_result_combined_list

#%%
import csv
csv_file = "/home/soumyajit/Codes/DL/100DaysOfML/Kaggle/Titanic/test_result.csv"
# with open(csv_file, "w") as output:
#     writer = csv.writer(output)
with open(csv_file, "w" , newline='') as output:
    writer = csv.writer(output)
    writer.writerow(["PassengerId","Survived"])
    writer.writerows(final_result_combined_list)

#the score is 0.73684 


