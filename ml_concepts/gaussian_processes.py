#!/usr/bin/env python3
'''
Gaussian processes
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

#
#
#

def synth_data(n_samples, sigma):
    X = np.random.uniform(low=0, high=2, size=n_samples)
    X = np.sort(X)
    y = np.sin(4*X) + np.random.normal(loc=0, scale=sigma, size=n_samples)
    return X, y

X_train, y_train = synth_data(n_samples=50, sigma=0.1)

plt.scatter(X_train, y_train, c='black')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

def kernel_rbf(x1, x2, l):
    Sigma = [[np.exp(-0.5*abs((x1[i]-x2[j])/l)**2) for j in range(len(x2))] for i in range(len(x1))]
    Sigma = np.array(Sigma)
    return Sigma

#
#
#

X_test = np.linspace(-0.5, 2.5, 200)
Sigma = kernel_rbf(X_test, X_test, l=1)

n_samples = 3
values = np.random.multivariate_normal([0]*len(X_test), Sigma, n_samples).T

colors = ['k','r','g','b','c','m','y','gray']*np.ceil(n_samples/8).astype(int)
for i in range(n_samples):
    plt.plot(X_test, values[:,i], c=colors[i], lw=2)

plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.show()

#
#
#

l, sigma_noise = 0.1, 0.5

k_xx = kernel_rbf(X_train, X_train, l)
k_xxs = kernel_rbf(X_train, X_test, l)
k_xsx = kernel_rbf(X_test, X_train, l)
k_xsxs = kernel_rbf(X_test, X_test, l)

fs = k_xsx.dot(np.linalg.inv(k_xx + sigma_noise**2 * np.identity(k_xx.shape[0]))).dot(y_train)
cov_fs = k_xsxs - k_xsx.dot(np.linalg.inv(k_xx + sigma_noise**2 * np.identity(k_xx.shape[0]))).dot(k_xxs)

# plot mean curve
plt.plot(X_test, fs, c='black')
plt.scatter(X_train, y_train, c='r', marker='+', lw=2)
plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.show()

# plot samples
n_samples = 3
values = np.random.multivariate_normal(fs, cov_fs, n_samples).T

colors = ['k','r','g','b','c','m','y','gray']*np.ceil(n_samples/8).astype(int)
for i in range(n_samples):
    plt.plot(X_test, values[:,i], c=colors[i], lw=2)

plt.scatter(X_train, y_train, c='b', marker='+', s=20*6)
plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.show()

#
#
#

# use train_test_split() twice
# start with original dataset as X, y
X, y = X_train, y_train

X_train, X_remain, y_train, y_remain = train_test_split(X, y, train_size=0.6)
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.2/(0.2+0.2))

plt.scatter(X_train, y_train, c='b', marker='+', lw=2, label='train')
plt.scatter(X_val, y_val, c='m', marker='+', lw=2, label='validation')
plt.scatter(X_test, y_test, c='r', marker='+', lw=2, label='test')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# optimize hyperparameters
sigma_values = np.linspace(0.3,1,20)
l_values = np.linspace(0.05,1,20)
mse = np.ones([20,20])*np.nan
mae = np.ones([20,20])*np.nan

for i, sigma_noise in enumerate(sigma_values):
    print(f'countdown:...{len(sigma_values)-i}')
    for j, l in enumerate(l_values):
        k_xx = kernel_rbf(X_train, X_train, l)
        k_xxs = kernel_rbf(X_train, X_val, l)
        k_xsx = kernel_rbf(X_val, X_train, l)
        k_xsxs = kernel_rbf(X_val, X_val, l)
        
        y_pred = fs = k_xsx.dot(np.linalg.inv(k_xx + sigma_noise**2 * np.identity(k_xx.shape[0]))).dot(y_train)
        cov_fs = k_xsxs - k_xsx.dot(np.linalg.inv(k_xx + sigma_noise**2 * np.identity(k_xx.shape[0]))).dot(k_xxs)
        
        mse[i,j] = ((y_pred-y_val)**2).mean()
        mae[i,j] = np.median(abs(y_pred-y_val))

row, col = np.unravel_index(mse.argmin(), mse.shape)

sigma_noise_opt = sigma_values[row]
l_opt = l_values[col]

rmse_value = np.sqrt(mse[row,col])
mae_value = mae[row,col]

df_error = pd.DataFrame([[rmse_value, mae_value]], columns=['rmse','mae'], index=['validation_error'])
print(df_error)

# test set evaluation
k_xx = kernel_rbf(np.r_[X_train,X_val], np.r_[X_train,X_val], l_opt)
k_xxs = kernel_rbf(np.r_[X_train,X_val], X_test, l_opt)
k_xsx = kernel_rbf(X_test, np.r_[X_train,X_val], l_opt)
k_xsxs = kernel_rbf(X_test, X_test, l_opt)

y_pred = fs = k_xsx.dot(np.linalg.inv(k_xx + sigma_noise_opt**2 * np.identity(k_xx.shape[0]))).dot(np.r_[y_train,y_val])  # number of domain points
cov_fs = k_xsxs - k_xsx.dot(np.linalg.inv(k_xx + sigma_noise_opt**2 * np.identity(k_xx.shape[0]))).dot(k_xxs)

rmse_value = np.sqrt(((y_pred-y_test)**2).mean())
mae_value = np.median(abs(y_pred-y_test))

df_error = pd.DataFrame([[rmse_value, mae_value]], columns=['rmse','mae'], index=['test_error'])
print(df_error)

#
#
#

# final model trained over all data
X_test = np.linspace(-0.5,2.5,200)  # reset test set to original one

k_xx = kernel_rbf(X, X, l_opt)
k_xxs = kernel_rbf(X, X_test, l_opt)
k_xsx = kernel_rbf(X_test, X, l_opt)
k_xsxs = kernel_rbf(X_test, X_test, l_opt)

y_pred = fs = k_xsx.dot(np.linalg.inv(k_xx + sigma_noise_opt**2 * np.identity(k_xx.shape[0]))).dot(y)
cov_fs = k_xsxs - k_xsx.dot(np.linalg.inv(k_xx + sigma_noise_opt**2 * np.identity(k_xx.shape[0]))).dot(k_xxs)

# plot mean curve
CI = 1.96*np.sqrt(np.diag(cov_fs))

plt.plot(X_test, y_pred, c='k')
plt.plot(X_test, y_pred-CI, c='r')
plt.plot(X_test, y_pred+CI, c='r')
plt.scatter(X, y, c='b', marker='+', lw=2)
plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.show() 

# plot samples
n_samples = 10
values = np.random.multivariate_normal(fs, cov_fs, n_samples).T

colors = ['k','r','g','b','c','m','y','gray']*np.ceil(n_samples/8).astype(int)
for i in range(n_samples):
    plt.plot(X_test, values[:,i], c=colors[i], alpha=0.6)

plt.scatter(X_train, y_train, c='k', marker='+', s=20*6)
plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.show()

#
#
#

# GP using sklearn
kernel = RBF()  # i.e. 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.05, 1))
y_pred_prior = GaussianProcessRegressor(kernel=kernel).sample_y(X_test.reshape(-1,1), n_samples=n_samples)  # sample from pior
model = GaussianProcessRegressor(kernel=kernel, alpha=0.5).fit(X.reshape(-1,1), y)  # alpha is sigma_noise (should be set)
# note: if alpha is set to sigma_noise_opt, we will obtain very similar results as manual implementation.
y_pred_posterior = model.sample_y(X_test.reshape(-1,1), n_samples=n_samples)  # sample from posterior
model.score(X.reshape(-1,1), y)
y_pred, sd = model.predict(X_test.reshape(-1,1), return_std=True)
model.kernel_  # show optimal parameters

# plot mean curve
plt.plot(X_test, y_pred, c='k')
plt.plot(X_test, y_pred-1.96*sd, c='r')
plt.plot(X_test, y_pred+1.96*sd, c='r')
plt.scatter(X, y, c='b', marker='+', s=20*6)
plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.show()

# plot samples
colors = ['k','r','g','b','c','m','y','gray']*np.ceil(n_samples/8).astype(int)
for i in range(n_samples):
    plt.plot(X_test, y_pred_posterior[:,i], c=colors[i], alpha=0.6)

plt.scatter(X_train, y_train, c='k', marker='+', s=20*6)
plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.show()
