#!/usr/bin/env python3
'''
feature selection & extraction
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from geneticalgorithm import geneticalgorithm as ga  # https://pypi.org/project/geneticalgorithm/

#
# 
#

iris = datasets.load_iris()
meas = iris.data
y = iris.target
X = np.random.multivariate_normal([0]*10, np.diag([1]*10), 150)
X = pd.DataFrame(X, columns=[f'V{i}' for i in range(10)])

X[['V0','V2','V4','V6']] = meas

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=4, step=1)
rfe.fit(X, y)
rfe.support_.nonzero()

#
#
#

path_to_file = './data/sonar.txt'
df = pd.read_csv(path_to_file, sep=' ')

pca = PCA()  # n_components not specified => keep all
pca.fit(df)
rv = pca.explained_variance_.cumsum()/pca.explained_variance_.sum()
pca.explained_variance_ratio_  # same as pca.explained_variance_/sum(pca.explained_variance_)
V = pca.components_  # components are on rows
pca.fit_transform(df)

plt.plot(range(1,60+1), rv, color='blue', linewidth=2,)
plt.xlabel('number of features')
plt.ylabel('ratio of retained variance to total variance')
plt.show()

#
#
#

df_scaled = StandardScaler().fit_transform(df)

pca = PCA().fit(df_scaled)
rv_scaled = pca.explained_variance_.cumsum()/pca.explained_variance_.sum()

plt.plot(range(1,60+1), rv, color='red', linewidth=2)
plt.plot(range(1,60+1), rv_scaled, color='black', linewidth=2)
plt.xlabel('number of features')
plt.ylabel('ratio of retained variance to total variance')
plt.legend(['original','normalized'])
plt.show()

#
#
#

iris = datasets.load_iris()
X = iris.data
y = iris.target
#target_names = iris.target_names

lda = LDA()
X_new = lda.fit(X, y).transform(X)
lda.fit(X, y).scalings_  # i.e. coefficients

#plt.scatter(X_new[:,0],X_new[:,1],color=pd.DataFrame(y).replace({0:'black', 1:'red', 2:'green'})[0])
plt.scatter(X_new[:,0], X_new[:,1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# w/ legend
'''
target_names = iris.target_names
for c, i, target_name in zip(['r','g','b'], [0, 1, 2], target_names):
    plt.scatter(X_new[y == i, 0], X_new[y == i, 1], color=c, label=target_name)

plt.legend()
plt.show()
'''

#
#
#

path_to_file = './data/circles.txt'
df = pd.read_csv(path_to_file, sep=' ')

# 3D plot
ax = plt.axes(projection='3d')
ax.scatter(df['f1'], df['f2'], df['f3'])
plt.show()

# PCA
df_pca = PCA(n_components=2).fit_transform(df)

plt.scatter(df_pca[:,0], df_pca[:,1], color='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

#
#
#

mds = MDS(n_components=2, max_iter=3000, dissimilarity="euclidean")
df_mds = mds.fit_transform(df)

plt.scatter(df_mds[:,0], df_mds[:,1], marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

#
#
#

path_to_file = './data/lines.txt'
df = pd.read_csv(path_to_file, sep=' ')

# 3D plot
ax = plt.axes(projection='3d')
ax.scatter(df['f1'], df['f2'], df['f3'])
plt.show()

# PCA
df_pca = PCA(n_components=2).fit_transform(df)

plt.scatter(df_pca[:,0], df_pca[:,1], color='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

#
#
#

mds = MDS(n_components=2, max_iter=3000, dissimilarity="euclidean")
df_mds = mds.fit_transform(df)

plt.scatter(df_mds[:,0], df_mds[:,1], marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

#
#
#

def Rastrigin(x):
    err = 20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
    return err

x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)

# reshape 1D array
e = np.array([Rastrigin([i,j]) for i in x1 for j in x2]).reshape(len(x1),len(x2))
e = pd.DataFrame(e, index=x1, columns=x2)

# surface plot
ax = plt.axes(projection="3d")
x, y = np.meshgrid(x2, x1)
ax.plot_surface(x, y, e)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# heatmap/contours
ticks_loc = np.arange(0, len(e.columns), 10)
plt.xticks(ticks=ticks_loc, labels=x1[ticks_loc].round(2), rotation=90)
plt.yticks(ticks=ticks_loc, labels=x2[ticks_loc].round(2))
plt.imshow(e, cmap='hot')
plt.colorbar()
plt.show()

#
#
#

varbound = np.array([[-3,3]]*2)
algorithm_param = {'max_num_iteration':1000,
                   'population_size':20,
                   'mutation_probability':0.2,
                   'elit_ratio':2/100,
                   'crossover_probability':0.8,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

model = ga(function=Rastrigin, dimension=2, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model.run()

# ... when there is a prevalence of local minima in the objective function.
# ... when it is difficult or impossible to implement gradient descent such as:
# ... when the error function is discontinuous, non-differentiable, stochastic, or highly nonlinear.
