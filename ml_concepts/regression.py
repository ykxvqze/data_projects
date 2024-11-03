#!/usr/bin/env python3
'''
regression
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#
#
#

def linear_regression(X, y):
    X = np.stack([X, np.ones(X.shape[0])], axis=1)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return beta

beta1, beta0 = -1, 2
sigma = 0.1

X = np.random.uniform(low=0, high=1, size=25)
noise = np.random.normal(loc=0, scale=sigma, size=25)
y = beta0 + beta1*X + noise

plt.scatter(X,y)
plt.show()

beta1_pred, beta0_pred = linear_regression(X,y)

# alternatively (scipy):
beta1_pred, beta0_pred, _, _, _ = linregress(X, y)

X_test = np.sort(X)
y_pred = beta0_pred + beta1_pred*X_test

# alternatively (sklearn):
model = LinearRegression().fit(X.reshape(-1,1), y)
beta1_pred = model.coef_
beta0_pred = model.intercept_
y_pred = model.predict(X_test.reshape(-1,1))

plt.scatter(X,y)
plt.plot(X_test, y_pred, c='r', linewidth=2)
plt.show()

#
#
#

size = [5,10,20,50,100]
b0_mean, b1_mean, b0_std, b1_std = ([] for i in range(4))

for i in size:
    b1, b0 = [], []
    for j in range(10):
        X = np.random.uniform(low=0, high=1, size=i)
        noise = np.random.normal(loc=0, scale=1, size=i)*sigma
        y = beta0 + beta1*X + noise
        beta1_pred, beta0_pred = linear_regression(X, y)
        b1.append(beta1_pred)
        b0.append(beta0_pred)
    b0_mean.append(np.mean(b0))
    b1_mean.append(np.mean(b1))
    b0_std.append(np.std(b0))
    b1_std.append(np.std(b1))

plt.errorbar(size, b0_mean, b0_std, label='beta0_pred')
plt.errorbar(size, b1_mean, b1_std, label='beta1_pred')
plt.xlabel('dataset size')
plt.ylabel('coefficients')
plt.legend()
plt.show()
# conclusion: variance of coefficient estimates decreases with dataset size;
# noise increases the variance of the estimates.

#
#
#

beta1, beta0 = -1, 2
sigma = 0.1

X = np.random.uniform(low=0, high=1, size=25)
noise = np.random.normal(loc=0, scale=sigma, size=25)
y = beta0 + beta1*X + noise

beta1_pred, beta0_pred = linear_regression(X,y)
X_test = np.sort(X)
y_pred = beta0_pred + beta1_pred*X_test

plt.scatter(X,y)
plt.plot(X_test, y_pred, c='r', linewidth=2)
plt.show()

SS_due_to_regression = ((y_pred - y.mean())**2).sum()
SS_about_the_mean = ((y - y.mean())**2).sum()
R_sq = SS_due_to_regression / SS_about_the_mean  # close to 1

# alternatively:
beta1_pred, beta0_pred, r_value, p_value, std_err = linregress(X, y)
R_sq = r_value**2

# alternatively:
model = LinearRegression().fit(X.reshape(-1,1), y)
beta1_pred = model.coef_
beta0_pred = model.intercept_
y_pred = model.predict(X_test.reshape(-1,1))
R_sq = model.score(X.reshape(-1,1), y)

# errorbar plot
coef = np.linspace(-1,1,9)
R_sq_mean, R_sq_std = [], []
for beta1 in coef:
    R_sq =[]
    for j in range(10):
        X = np.random.uniform(low=0, high=1, size=25)
        noise = np.random.normal(loc=0, scale=sigma, size=25)
        y = beta0 + beta1*X + noise
        
        beta1_pred, beta0_pred, r_value, p_value, std_err = linregress(X, y)
        R_sq.append(r_value**2)
    
    R_sq_mean.append(np.mean(R_sq))
    R_sq_std.append(np.std(R_sq))

plt.errorbar(coef, R_sq_mean, R_sq_std, linewidth=2)
plt.xlabel('beta1')
plt.ylabel('R_sq_mean')
plt.ylim([-0.5,1.5])
plt.show()

#
#
#

def kernel_regression(X_train, y_train, X_test, h):  # 1D case
    y_pred = []
    for i in range(len(X_test)):
        K = np.exp(-(X_train-X_test[i])**2/h**2)
        y_pred.append((K*y_train).sum()/np.sum(K))
    return y_pred

def synth_data(n_samples, sigma):
    X = np.random.uniform(low=0, high=1, size=n_samples)
    y = np.sin(4*X) + np.random.normal(loc=0, scale=sigma, size=n_samples)
    return X, y

X_train, y_train = synth_data(n_samples=50, sigma=0.1)

plt.scatter(X_train, y_train)
plt.show()

X_test, h = np.arange(0, 1, 0.01), 0.1
y_test = kernel_regression(X_train, y_train, X_test, h)

plt.scatter(X_train, y_train)
plt.plot(X_test, y_test, c='r', linewidth=2)
plt.show()
# a good value for h is 0.1: if it's lower => overfitting, if it's higher => underfitting.

#
#
#

def optimize_h(X, y, h):  #1D case
    rep = 5
    mse = np.zeros([rep,len(h)])
    for j, bandwidth in enumerate(h):
        for i in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            y_pred = kernel_regression(X_train, y_train, X_test, bandwidth)
            mse[i,j] = ((y_pred - y_test)**2).mean()
    plt.errorbar(h, mse.mean(axis=0), mse.std(axis=0), linewidth=2)
    plt.xlabel('h')
    plt.ylabel('mse')
    plt.show()

X_train, y_train = synth_data(n_samples=50, sigma=0.1)

optimize_h(X_train, y_train, [0.01,0.05,0.1,0.2,0.3])

#
#
#

path_to_file = './data/prostate.txt'
df = pd.read_csv(path_to_file, sep=' ')

corrmat = df.drop(['training_set'],axis=1).corr().round(2)

plt.xticks(ticks=np.arange(len(df.columns)-1), labels=df.columns[:-1],rotation=90)
plt.yticks(ticks=np.arange(len(df.columns)-1), labels=df.columns[:-1])
plt.imshow(corrmat, cmap='hot')
plt.colorbar()
plt.show()
# features 1,5,6 seem mostly correlated with the target variable.

#
#
#

X = df[df['training_set']==1].drop(['target','training_set'],axis=1)
y = df[df['training_set']==1]['target']

X = (X - X.mean())/X.std()
n, p = X.shape

L = [0.1,1,10,20,50,100,200,500,1000,2000,5000,10000]

list_beta = [np.linalg.inv(X.T.dot(X) + np.identity(p)*l).dot(X.T).dot(y-y.mean()) for l in L]
beta = np.array(list_beta)

# import matplotlib.cm as cm
colors = ['k','r','g','b','c','m','y','gray']
for i in range(p):
    plt.plot(np.log10(L), beta[:,i], c=colors[i], linewidth=2, label='beta'+str(i+1))
    plt.scatter(np.log10(L), beta[:,i], c=colors[i], linewidth=2)

plt.plot(np.log10(L), np.zeros(len(L)), c='red', linewidth=2)
plt.xlabel('$log_{10}(\lambda$)')
plt.ylabel('coefficients')
plt.legend()
plt.grid()
plt.show()

#
#
#

# heatmap
plt.xticks(ticks=range(len(L)), labels=L, rotation=90)
plt.yticks(ticks=range(p), labels=['beta'+str(i+1) for i in range(p)])
plt.imshow(beta.T,cmap='hot') #(8, 12)
plt.xlabel('$\lambda$')
plt.colorbar()

# (optional) add values as text in heatmap: loop over data dimensions and create text annotations
for i in range(p):
    for j in range(len(L)):
        plt.text(j, i, beta[j, i].round(2), ha="center", va="center", c="k")

plt.show()
