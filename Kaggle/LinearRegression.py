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
#The equivalent implementation of the above Algorithm
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)
