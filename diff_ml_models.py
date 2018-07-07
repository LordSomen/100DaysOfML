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

