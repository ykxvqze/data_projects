#!/usr/bin/env python3
'''
fundamental concepts
'''

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

#
#
#

x = np.random.normal(loc=0, scale=1, size=1000)
nb = 10
plt.hist(x, bins=nb, density=False)
plt.xlabel('x')
plt.ylabel('Frequency')
plt.show()

#
#
#

# Rank statistics are more robust to outliers.

#
#
#

y = np.random.normal(loc=0, scale=1, size=10000)
y.mean()
y.var()

fig, ax = plt.subplots(1,2)
ax[0].hist(y, density=False)
ax[0].set_xlabel('y')
ax[0].set_ylabel('Frequency')

ax[1].boxplot(y, patch_artist=True, vert=True)
ax[1].set_xlabel('y')
plt.show()

#
#
#

b = np.exp(-0.5*y)
b.mean()
b.var()

fig, ax = plt.subplots(1,2)
ax[0].hist(b, density=False)
ax[0].set_xlabel('b')
ax[0].set_ylabel('Frequency')

ax[1].boxplot(b, patch_artist=False, vert=True)
ax[1].set_xlabel('b')
plt.show()

#
#
#

path_to_file = './data/boomerang3D.txt'
df = pd.read_csv(path_to_file, sep=' ')

ax = plt.axes(projection='3d')
ax.scatter(df['feature1'], df['feature2'], df['feature3'], c=df['label'])
# choose colors
#colors = np.repeat(['blue','red'], 500)
#ax.scatter(df['feature1'], df['feature2'], df['feature3'], c=colors)
ax.set_xlabel('feature1')
ax.set_ylabel('feature2')
ax.set_zlabel('feature3')
plt.show()

# to add a legend, create a scatter plot for each class
df1 = df.loc[df['label']==1]
df2 = df.loc[df['label']==2]

ax = plt.axes(projection='3d')
scatter1 = ax.scatter(df1['feature1'], df1['feature2'], df1['feature3'], c='blue')
scatter2 = ax.scatter(df2['feature1'], df2['feature2'], df2['feature3'], c='red')
ax.legend([scatter1, scatter2], ['label=1', 'label=2'])
plt.show()

# 2D scatter plot
plt.scatter(df['feature1'], df['feature2'], c=df['label'])
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.show()

#
#
#

path_to_file = './data/dataset8D.txt'
df = pd.read_csv(path_to_file, sep=' ')
df.info()

scatter_matrix(df.iloc[:,:-1], diagonal='kde')
plt.show()

covmat = np.cov(df.iloc[:,:-1].T).round(2)
#np.cov(df.iloc[:,:-1], rowvar=False).round(2)

corrmat = df.drop(['label'], axis=1).corr().round(2)
#corrmat = np.corrcoef(df.drop(['label'], axis=1), rowvar=False).round(2)

plt.xticks(ticks=range(len(corrmat.columns)), labels=corrmat.columns, rotation=90)
plt.yticks(ticks=range(len(corrmat.columns)), labels=corrmat.columns)
plt.imshow(corrmat, cmap='hot')
plt.colorbar()
plt.show()

#
#
#

x = np.random.uniform(low=0, high=1, size=100)
x.mean()  # theoretical: (b-a)/2 = 1/2 = 0.5
x.var()  # theoretical: (b-a)^2/12 = 1/12 = 0.08333

mu = []
for i in range(1000):
    x = np.random.uniform(low=0, high=1, size=100)
    mu.append(x.mean())

mu = np.array(mu)
mu.mean()
mu.var()  # theoretical: 0.08333/(n=100)

print(f'mean of mu: \t {np.mean(mu)}')
print(f'variance of mu: \t {np.var(mu)}')

plt.hist(mu, density=False)
plt.show()

#
#
#

n = [10, 100, 250, 500, 1000]
mean_m, mean_v, sd_m, sd_v = ([] for i in range(4));

for i in n:
    m,v = [],[]
    for j in range(1000):
        x = np.random.normal(loc=0, scale=1, size=i)
        m.append(x.mean())
        v.append(x.var())
    mean_m.append(np.mean(m))
    sd_m.append(np.std(m))
    mean_v.append(np.mean(v))
    sd_v.append(np.std(v))

plt.errorbar(n, mean_m, sd_m)
plt.xlabel('n')
plt.ylabel('mean_m')
plt.show()

plt.errorbar(n, mean_v, sd_v)
plt.xlabel('n')
plt.ylabel('mean_v')
plt.show()

#
#
#

mean = np.zeros(2)
cov = np.eye(2)
x = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)
S = np.cov(x, rowvar=False)
w, v = np.linalg.eig(S)

#
#
#

x = np.random.multivariate_normal(mean=[0,0], cov=[[3,0],[0,1]], size=1000)
S = np.cov(x.T)
w, v = np.linalg.eig(S)

plt.scatter(x[:,0], x[:,1])
plt.plot([0, v[0,0]], [0, v[1,0]], color='red', linewidth=2)
plt.plot([0, v[0,1]], [0, v[1,1]], color='red', linewidth=2)
plt.axis('equal')
plt.show()

plt.scatter(x[:,0], x[:,1])
plt.plot(np.sqrt(w[0])*np.array([0, v[0,0]]), np.sqrt(w[0])*np.array([0, v[1,0]]), color='red', lw=2)
plt.plot(np.sqrt(w[1])*np.array([0, v[0,1]]), np.sqrt(w[1])*np.array([0, v[1,1]]), color='red', lw=2)
plt.axis('equal')
plt.show()

#
#
#

x = np.random.multivariate_normal(mean=[0,0], cov=[[3,-2],[-2,2]], size=1000)
S = np.cov(x.T)
w, v = np.linalg.eig(S)

plt.scatter(x[:,0], x[:,1])
plt.plot(np.sqrt(w[0])*np.array([0, v[0,0]]), np.sqrt(w[0])*np.array([0, v[1,0]]), color='red', linewidth=2)
plt.plot(np.sqrt(w[1])*np.array([0, v[0,1]]), np.sqrt(w[1])*np.array([0, v[1,1]]), color='red', linewidth=2)
plt.axis('equal')
plt.show()

#
#
#

z = np.random.multivariate_normal(mean=[0,0], cov=[[3,-2],[-2,2]], size=1000)
S = np.cov(z.T)
w, v = np.linalg.eig(S)

D = np.diag(w)
w = z.dot(v).dot(np.linalg.inv(np.sqrt(D)))

plt.scatter(z[:,0], z[:,1], color='blue', label= 'original')
plt.scatter(w[:,0], w[:,1], color='red', label= 'sphered')
plt.axis('equal')
plt.legend()
plt.show()

#
#
#

path_to_file = './data/dataset1D.txt'
df = pd.read_csv(path_to_file, sep=' ')

# pandas' kde method
a = df['a']
a.plot.kde(bw_method=0.1)
plt.scatter(a, np.zeros(len(a)), color='red', marker='+')
plt.xlabel('a')
plt.ylabel('density')
plt.show()

# sklearn's KernelDensity() method
model = KernelDensity(kernel='gaussian', bandwidth=1)
x = np.linspace(a.min()-3, a.max()+3, 1000).reshape(-1,1)
model.fit(np.array(a).reshape(-1,1))
scores_log = model.score_samples(x)
scores = np.exp(scores_log)

plt.plot(x, scores, linewidth=2)
plt.scatter(a, np.zeros(len(a)), color='red', marker='+')
plt.xlabel('a')
plt.ylabel('density')
plt.show()

#
#
#

h = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5]
ll = []
for i in h:
    model = KernelDensity(bandwidth=i)
    model.fit(np.array(df['trn']).reshape(-1,1))
    scores_log = model.score_samples(np.array(df['tst']).reshape(-1,1))
    ll.append(scores_log.sum())

plt.plot(h, ll, color='blue', linewidth=2)
plt.scatter(h, ll, color='blue')
plt.xlabel('h')
plt.ylabel('LL')
plt.show()

h[np.array(ll).argmax()]

#
#
#

def knnd(a, b, k=1):
    n = len(a)
    D = cdist(b, a)
    D.sort(axis=1)  # in-place
    density = (k/n)/D[:,(k-1)]
    return density

path_to_file = './data/dataset1D.txt'
df = pd.read_csv(path_to_file, sep=' ')

a = np.array(df['a']).reshape(-1,1)  # entire dataset
b = np.arange(min(a)-1, max(a)+1, 0.1).reshape(-1,1)  # grid

density = knnd(a, b, k=3)
plt.plot(b, density)
plt.xlabel('a')
plt.ylabel('density, k=3')
plt.ylim([-0.2, max(density)])
plt.scatter(a, np.zeros(len(a)), color='red', marker='+')
plt.show()
