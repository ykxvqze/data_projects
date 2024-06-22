#!/usr/bin/env python3
'''
unsupervised learning
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, sklearn
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn import datasets
from minisom import MiniSom  # from: https://github.com/JustGlowing/minisom
from fcmeans import FCM  # from: https://pypi.org/project/fuzzy-c-means/

print("Python Version : ", sys.version)
print("Scikit-Learn Version : ", sklearn.__version__)

#
#
#

path_to_file = './data/clust14.txt'
X = pd.read_csv(path_to_file, sep=' ')

plt.scatter(X['f1'], X['f2'], color='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

Z = linkage(X, method='complete')  # linkage matrix
dendrogram(Z, no_labels=True)
plt.show()

model = AgglomerativeClustering(n_clusters=14, linkage='complete')
model.fit(X)
model.labels_

plt.scatter(X['f1'], X['f2'], c=model.labels_, cmap=plt.cm.jet, marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

# scipy: "The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]."
Fc = Z[::-1,2]

plt.plot(range(1,16+1), Fc[:16], c='blue', linewidth=2)
plt.scatter(range(1,16+1), Fc[:16], c='blue', linewidth=2)
plt.xlabel('number of clusters')
plt.ylabel('height')
plt.show()

model = AgglomerativeClustering(n_clusters=3, linkage='complete')
model.fit(X)
model.labels_

plt.scatter(X['f1'], X['f2'], c=model.labels_, cmap=plt.cm.jet, marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

path_to_file = './data/parallelclust.txt'
X = pd.read_csv(path_to_file, sep=' ')

plt.scatter(X['f1'], X['f2'], c='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# complete linkage
Z = linkage(X, method='complete')
dendrogram(Z, no_labels=True)
plt.show()

model = AgglomerativeClustering(n_clusters=2, linkage='complete')
model.fit(X)
model.labels_

plt.scatter(X['f1'], X['f2'], c=model.labels_, cmap=plt.cm.jet, marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# single linkage
Z = linkage(X, method='single')
dendrogram(Z, no_labels=True)
plt.show()

model = AgglomerativeClustering(n_clusters=2, linkage='single')
model.fit(X)
model.labels_

plt.scatter(X['f1'], X['f2'], c=model.labels_, cmap=plt.cm.jet, marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

path_to_file = './data/gaussianclust.txt'
X = pd.read_csv(path_to_file, sep=' ')

plt.scatter(X['f1'], X['f2'], c='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# complete linkage
Z = linkage(X, method='complete')
dendrogram(Z, no_labels=True)
plt.show()

model = AgglomerativeClustering(n_clusters=2, linkage='complete')
model.fit(X)
model.labels_

plt.scatter(X['f1'], X['f2'], c=model.labels_, cmap=plt.cm.jet, marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

path_to_file = './data/clust3.txt'
X = pd.read_csv(path_to_file, sep=' ')

plt.scatter(X['f1'], X['f2'], c='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

indices = []
for i in range(2,10+1):
    model = AgglomerativeClustering(n_clusters=i, linkage='complete').fit(X)
    indices.append(davies_bouldin_score(X, model.labels_))

plt.plot(range(2,10+1), indices, 'b-o', linewidth=2)
plt.xlabel('number of clusters')
plt.ylabel('Davies-Bouldin index')
plt.show()

#
#
#

path_to_file = './data/gaussianclust.txt'
X = pd.read_csv(path_to_file, sep=' ')

plt.scatter(X['f1'], X['f2'], c='blue',marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

model = KMeans(n_clusters=2, max_iter=100, n_init=10).fit(X)
model.labels_
model.cluster_centers_  # each row is a centroid

plt.scatter(X['f1'], X['f2'], c=model.labels_, marker='+')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='magenta', linewidth=4, marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

path_to_file = './data/clust3.txt'
X = pd.read_csv(path_to_file, sep=' ')

plt.scatter(X['f1'], X['f2'], c='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

model = KMeans(n_clusters=3, max_iter=100, n_init=10).fit(X)
model.labels_
model.cluster_centers_

plt.scatter(X['f1'], X['f2'], c=model.labels_, marker='+')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='magenta', linewidth=4, marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

path_to_file = './data/parallelclust.txt'
X = pd.read_csv(path_to_file, sep=' ')

plt.scatter(X['f1'], X['f2'], c='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

model = KMeans(n_clusters=2, max_iter=100, n_init=10).fit(X)
model.labels_
model.cluster_centers_

plt.scatter(X['f1'], X['f2'], c=model.labels_, marker='+')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='magenta', linewidth=4, marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

path_to_file = './data/clust3.txt'
X = pd.read_csv(path_to_file, sep=' ')

model = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X)
labels = mixture.GaussianMixture(n_components=3, covariance_type='full').fit_predict(X)

plt.scatter(X['f1'], X['f2'], c=labels, marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

path_to_file = './data/parallelclust.txt'
X = pd.read_csv(path_to_file, sep=' ')

labels = mixture.GaussianMixture(n_components=2, covariance_type='full').fit_predict(X)

plt.scatter(X['f1'], X['f2'], c=labels, marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#
#
#

path_to_file = './data/clust3.txt'
X = pd.read_csv(path_to_file, sep=' ')

bic_model = []
for i in range(1,9+1):
    model = mixture.GaussianMixture(n_components=i, covariance_type='full').fit(X)
    bic_model.append(model.bic(X))

plt.plot(range(1,9+1), bic_model, 'b-o')
plt.xlabel('number of components')
plt.ylabel('BIC')
plt.show()

#
#
#

iris = datasets.load_iris()
X = iris.data
y = iris.target

plt.scatter(X[:,2], X[:,3], c='blue', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

n_neurons = 10
m_neurons = 10
X = X[:,-2:]
som = MiniSom(n_neurons, m_neurons, X.shape[1], sigma=1.5, learning_rate=0.05, neighborhood_function='gaussian')

som.pca_weights_init(X)
som.train(X, 5000, verbose=True)

plt.imshow(som.distance_map().T, cmap='hot')  # heatmap
plt.colorbar()
plt.show()

#
#
#

path_to_file = './data/cust3.txt'
X = pd.read_csv(path_to_file, sep=' ')

plt.scatter(X['f1'],X['f2'],c='blue',marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

n_neurons = 10
m_neurons = 10
som = MiniSom(n_neurons, m_neurons, X.shape[1], sigma=1.5, learning_rate=0.05, neighborhood_function='gaussian')

som.pca_weights_init(X.to_numpy())  # or X.values
som.train(X.to_numpy(), 5000, verbose=True)

plt.imshow(som.distance_map().T, cmap='hot')
plt.colorbar()
plt.show()

#
#
#

path_to_file = './data/tissue.png'
image = plt.imread(path_to_file)
plt.imshow(image)
plt.show()

R = image[:,:,0].ravel()
G = image[:,:,1].ravel()
B = image[:,:,2].ravel()

X = np.stack([R,G,B], axis=1)

fcm = FCM(n_clusters=4, max_iter=100, m=2)
fcm.fit(X)

# outputs
fcm_centers = fcm.centers
fcm_memberships = fcm.u
fcm_labels  = fcm.u.argmax(axis=1)

pmap0 = fcm_memberships[:,0].reshape(image.shape[0:2])
pmap1 = fcm_memberships[:,1].reshape(image.shape[0:2])
pmap2 = fcm_memberships[:,2].reshape(image.shape[0:2])
pmap3 = fcm_memberships[:,3].reshape(image.shape[0:2])

pmap0_pseudo_colored = image * np.repeat(pmap0, 3).reshape(pmap0.shape[0], pmap0.shape[1], 3)
pmap1_pseudo_colored = image * np.repeat(pmap1, 3).reshape(pmap1.shape[0], pmap1.shape[1], 3)
pmap2_pseudo_colored = image * np.repeat(pmap2, 3).reshape(pmap2.shape[0], pmap2.shape[1], 3)
pmap3_pseudo_colored = image * np.repeat(pmap3, 3).reshape(pmap3.shape[0], pmap3.shape[1], 3)

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(pmap0,cmap='gray')
ax[1].imshow(pmap0_pseudo_colored)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(pmap1,cmap='gray')
ax[1].imshow(pmap1_pseudo_colored)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(pmap2,cmap='gray')
ax[1].imshow(pmap2_pseudo_colored)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(pmap3,cmap='gray')
ax[1].imshow(pmap3_pseudo_colored)
plt.show()
