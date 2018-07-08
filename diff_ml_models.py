#%%
''' this is a implementation of the Linear regression model
from scratch without using any library '''
import numpy as np
X = 2* np.random.rand(100,1)
X

#%%
#the dependent equation i.e Y's value depends on Y
Y = 4 + 3 * X + np.random.randn(100, 1)
Y
#%%
#creates 100 X 1 shape matrix with value 1
np.ones((100,1))
#%%
X_b = np.c_[np.ones((100, 1)), X]
# add x0 = 1 to each instance
X_b

#%%
''' the normal equation'''
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y) 
theta_best

#%%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
y_predict

#%%
import matplotlib.pyplot as plt
plt.plot(X_new, y_predict, "r-")
plt.plot(X, Y, "g.")
plt.axis([0, 2, 0, 15])
plt.show()

#%%
#The equivalent implementation of the above Algorithm with library 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

#%%
''' this is the implementation of the gradient descent algorithm 
from scratch'''
eta = 0.1 # learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)
theta

#%%
for iteration in range(n_iterations):
    gradient = 2/m * X_b.T.dot(X_b.dot(theta) - Y)
    theta = theta - eta * gradient
theta

#%%
''' 
this is the implementation of the stochastic gradient
descent algorithm
'''

epochs = 50 
t0 , t1 = (5,50)

def learning_rate(t):
    return t0 / (t + t1)

for epoch in range(epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        x_i = X_b[random_index:(random_index + 1)]
        y_i = Y[random_index:random_index+1]
        gradients = 2 * x_i.T.dot(x_i.dot(theta) - y_i)
        eta = learning_rate(epoch * m + i)
        theta = theta - eta * gradients

theta

#%%
from sklearn.preprocessing import PolynomialFeatures
m = 100
X_p = 6 * np.random.rand(m, 1) - 3
Y_p = 0.5 * X_p**2 + X_p + 2 + np.random.randn(m, 1)
poly_features = PolynomialFeatures(degree=2 , include_bias=False)
X_poly = poly_features.fit_transform(X_p)
X_p[0]
X_poly[0]

#%%
lin_reg = LinearRegression()
lin_reg.fit(X_poly,Y_p)
print(lin_reg.intercept_ , lin_reg.coef_)

#%%
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model,X,Y):
    X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2)
    train_error,val_error = [],[]
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],Y_train[:m])
        Y_train_predict = model.predict(X_train[:m])
        Y_val_predict = model.predict(X_val)
        train_error.append(mean_squared_error(Y_train[:m],Y_train_predict))
        val_error.append(mean_squared_error(Y_val,Y_val_predict))
    plt.plot(np.sqrt(train_error), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_error), "b-", linewidth=3, label="val")

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X_p, Y_p)

#%%
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((
("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
("sgd_reg", LinearRegression()),
))
plot_learning_curves(polynomial_regression, X_p, Y_p)



