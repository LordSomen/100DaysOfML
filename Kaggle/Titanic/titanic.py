# %%
import pandas as pd

passenger_train = pd.read_csv("train.csv", sep=",")
passenger_train

# %%
passenger_test = pd.read_csv("test.csv")
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
passenger_train_var["Age_Per_Fare"] = passenger_train_var["Age"]
    /passenger_train_var["Fare"]
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

