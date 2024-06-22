#!/usr/bin/env python3
'''
supervised learning 1
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

#
#
#

# Bayes' theorem and p(\omega_2|x) = 1 - p(\omega_1|x)

#
#
#

path_to_file = './data/gaussian2D.txt'
df = pd.read_csv(path_to_file, sep=' ')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

model = LogisticRegression(solver='lbfgs').fit(X, y)
y_pred = model.predict(X)
confusion_matrix(y_true=y, y_pred=y_pred)
error_apparent = (y!=y_pred).mean()

# sigmoid surface plot
w_0 = model.intercept_
w = model.coef_

x1 = np.linspace(-5,15,200)
x2 = np.linspace(-5,15,200)
x, y = np.meshgrid(x1, x2)

z = [[np.exp(w_0 + w.dot([i,j]))[0] / (1 + np.exp(w_0 + w.dot([i,j])))[0] for i in x1] for j in x2]
z = np.array(z)

ax = plt.axes(projection="3d")
ax.plot_surface(x, y, z, cmap='jet')
ax.scatter(X['f1'], X['f2'], np.zeros(X.shape[0])+0.1, c='red')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

#
#
#

path_to_trainfile = './data/trainfile.txt'
path_to_testfile = './data/testfile.txt'

a = pd.read_csv(path_to_trainfile, sep=' ', header=None)
b = pd.read_csv(path_to_testfile, sep=' ', header=None)

X_train, y_train = a.iloc[:,:-1], a.iloc[:,-1]
X_test, y_test = b.iloc[:,:-1], b.iloc[:,-1]

plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

m1 = X_train.loc[y_train==1].mean()
m2 = X_train.loc[y_train==2].mean()

S1 = np.cov(X_train.loc[y_train==1], rowvar=False)
S2 = np.cov(X_train.loc[y_train==2], rowvar=False)

n_obj, n_dim = X_test.shape

np.tile(m1,(n_obj,1))

X_test_bar1 = X_test - np.tile(m1, (n_obj,1))
X_test_bar2 = X_test - np.tile(m2, (n_obj,1))

# instead of looping over each object (row of X_test), use the full matrix, X_test, and consider the resulting diagonal elements only.
prob1 = 1/np.sqrt((2*np.pi)**n_dim*np.linalg.det(S1)) * np.diag(np.exp(-0.5*(X_test_bar1).dot(np.linalg.inv(S1)).dot(X_test_bar1.T)))
prob2 = 1/np.sqrt((2*np.pi)**n_dim*np.linalg.det(S2)) * np.diag(np.exp(-0.5*(X_test_bar2).dot(np.linalg.inv(S2)).dot(X_test_bar2.T)))

prior1, prior2 = 0.5, 0.5
px = prob1*prior1 + prob2*prior2

posterior1 = prob1*prior1/px
posterior2 = prob2*prior2/px
posterior = np.stack([posterior1, posterior2], axis=1)

posterior.sum(axis=1)
y_pred = 1 + posterior.argmax(axis=1)

n_error = (y_test!=y_pred).sum()
confusion_matrix(y_true=y_test, y_pred=y_pred)

#
#
#

# ...

#
#
#

model = qda().fit(X_train, y_train)
y_pred = model.predict(X_test)
confusion_matrix(y_true=y_test, y_pred=y_pred)

# decision boundary
X = pd.concat([X_train, X_test])  # full dataset to derive axes limits
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

labels_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4)
#plt.contour(xx, yy, Z, alpha=.6, cmap='gray')  # optional contour
#Zlines = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # contour lines
#plt.contour(xx, yy, Zlines, cmap=plt.cm.Paired)
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train.replace({1:'black', 2:'red'}))
plt.show()

#
#
#

model = lda().fit(X_train, y_train)
y_pred = model.predict(X_test)
confusion_matrix(y_true=y_test, y_pred=y_pred)

# decision boundary
X = pd.concat([X_train, X_test])
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

labels_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4)
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train.replace({1:'black', 2:'red'}))
plt.show()

#
#
#

path_to_file = './data/parallelclasses.txt'
df = pd.read_csv(path_to_file, sep=' ')
X, y = df.iloc[:,:-1], df.iloc[:,-1]

model = lda().fit(X, y)
y_pred = model.predict(X)
confusion_matrix(y_true=y, y_pred=y_pred)
error_apparent = (y_pred!=y).sum()

# decision boundary
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

labels_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4)
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y.replace({1:'black', 2:'red'}))
plt.show()

#
#
#

path_to_file = './data/parallelclasses.txt'
df = pd.read_csv(path_to_file, sep=' ')
X, y = df.iloc[:,:-1], df.iloc[:,-1]

model = qda().fit(X, y)
y_pred = model.predict(X)
confusion_matrix(y_true=y, y_pred=y_pred)
error_apparent = (y_pred!=y).sum()

# decision boundary
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

labels_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4)
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y.replace({1:'black', 2:'red'}))
plt.show()

#
#
#

path_to_file = './data/boomerang2D.txt'
df = pd.read_csv(path_to_file, sep=' ')
X, y = df.iloc[:,:-1], df.iloc[:,-1]

model = qda().fit(X, y)
y_pred = model.predict(X)
confusion_matrix(y_true=y, y_pred=y_pred)
error_apparent = (y_pred!=y).sum()

# decision boundary
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

labels_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4)
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y.replace({1:'black', 2:'red'}))
plt.show()

#
#
#

path_to_trainfile = './data/trainfile.txt'
path_to_testfile = './data/testfile.txt'

a = pd.read_csv(path_to_trainfile, sep=' ', header=None)
b = pd.read_csv(path_to_testfile, sep=' ', header=None)

X_train, y_train = a.iloc[:,:-1], a.iloc[:,-1]
X_test, y_test = b.iloc[:,:-1], b.iloc[:,-1]

model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = model.predict(X_test)

confusion_matrix(y_true=y_test, y_pred=y_pred)
error_apparent = (y_pred!=y_test).sum()

# decision boundary
X = pd.concat([X_train, X_test])
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

labels_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4)
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train.replace({1:'black', 2:'red'}))
plt.show()

#
#
#

dimensionality = [2,3,5,10,25,100]
error_test_mean, error_test_std = [],[]

for i in dimensionality:
    error_test = []
    for j in range(10):
        X1 = np.random.multivariate_normal(mean=np.zeros(i), cov=np.identity(i), size=10+1000)
        X2 = np.random.multivariate_normal(mean=np.append(2, np.zeros(i-1)), cov=np.identity(i), size=10+1000)
        
        X_train = np.concatenate([X1[:10,:], X2[:10,:]], axis=0)
        X_test = np.concatenate([X1[10:,:], X2[10:,:]], axis=0)
        y_train = np.repeat([1,2], 10)
        y_test = np.repeat([1,2], 1000)
        
        model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error_test.append((y_pred!=y_test).mean())
        
    error_test_mean.append(np.mean(error_test))
    error_test_std.append(np.std(error_test))

plt.errorbar(dimensionality, error_test_mean, error_test_std)
plt.scatter(dimensionality, error_test_mean)
plt.xlabel('dimensionality')
plt.ylabel('test error')
plt.show()

#
#
#

def parzenc(X_train, y_train, X_test, h):
    classes = y_train.unique()
    priors = y_train.value_counts()[classes].values/len(y_train)
    testDistMat = distance.cdist(X_test, X_train, 'euclidean')
    g = 1/(h*np.sqrt(2*np.pi))*np.exp(-0.5*(testDistMat/h)**2)
    p = np.zeros((X_test.shape[0], len(classes)))
    
    for i in range(len(classes)):
        tmp = g[:, y_train==classes[i]]
        p[:,i] = tmp.mean(axis=1)
    
    posteriors = priors*p/((priors*p).sum(axis=1, keepdims=True))
    labels = classes[posteriors.argmax(axis=1)]
    
    return posteriors, labels

path_to_file = './data/boomerang2D.txt'
df = pd.read_csv(path_to_file, sep=' ')
X, y = df.iloc[:,:-1], df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

y_pred_prob, y_pred = parzenc(X_train, y_train, X_test, h=0.5)
error_test = (y_pred!=y_test).sum()

# decision boundary
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

_ , labels_pred = parzenc(X, y, np.c_[xx.ravel(), yy.ravel()], h=0.5)
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4)
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y.replace({1:'black', 2:'red'}))
plt.show()
