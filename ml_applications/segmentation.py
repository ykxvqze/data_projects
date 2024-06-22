#!/usr/bin/env python3
'''
customer segmentation
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#
# data processing
#

path_to_file = './data/invoices.csv'
data = pd.read_csv(path_to_file)
data.head()
data.shape

data.loc[data['quantity']<0].head(6)
df = data.loc[data['quantity']>0]
df.loc[df['customer_id'].isna()]
df = df.dropna(subset=['customer_id'])
df.shape

df['invoice_date'] = pd.to_datetime(df['invoice_date'])
df['invoice_date'].min()
df['invoice_date'].max()

df = df.loc[df['invoice_date']<'2011-12-01']
df.shape

df['quantity'].value_counts()

#
# features
#

df['sales'] = df['quantity'] * df['unit_price']

df_customers = df.groupby(['customer_id']).aggregate({'sales': 'sum', 'invoice_date': 'nunique'})
df_customers = df_customers.rename(columns={'sales':'total_sales', 'invoice_date':'order_count'})
df_customers['avg_sales'] = df_customers['total_sales']/df_customers['order_count']

df_customers.describe()
df_customers.shape

df_rank = df_customers.rank(method='first')

df_norm = pd.DataFrame(scale(df_rank), index=df_rank.index, columns=df_rank.columns)

df_norm.describe().loc[['mean','std']]

#
# customer segmentation (k-means clustering)
#

model = KMeans(n_clusters=4, max_iter=100, n_init=10).fit(df_norm)
model.labels_
model.cluster_centers_  # each row => centroid

values, counts = np.unique(model.labels_, return_counts=True)
dict(zip(values, counts))

df_customers['label'] = model.labels_
df_customers.groupby(['label']).mean()

# optimal number of clusters by silhouette
k_optimal, highest_score = 0, -1
for k in range(4,11):
    labels = KMeans(n_clusters=k, max_iter = 100, n_init=10).fit_predict(df_norm)
    score = silhouette_score(df_norm, labels)
    print(f'For k={k}, silhouette coefficient = {score}')
    
    if highest_score < score:
        highest_score = score
        k_optimal = k

k_optimal

# alternatively:
k_values, scores = range(4,11), []
for k in k_values:
    model = KMeans(n_clusters=k, max_iter = 100, n_init=10).fit(df_norm)
    score = silhouette_score(df_norm, model.labels_)
    scores.append(score)

k_optimal = k_values[np.argmax(scores)]
k_optimal

model = KMeans(n_clusters=k_optimal, max_iter = 100, n_init=10).fit(df_norm)

# overwrite labels with those of final model
df_customers['label'] = model.labels_

table = df_customers.groupby('label').mean()
high_value_cluster = table['avg_sales'].idxmax()

df_norm['label'] = model.labels_

# identify high-valued segment
high_valued_customers = df_customers.loc[df_customers['label'] == high_value_cluster]
high_valued_customers.describe()

vip = np.array(high_valued_customers.index)

df_vip = df.loc[df['customer_id'].isin(vip)]

items = df_vip.groupby('description').aggregate({'description': 'count'})
items = items.rename(columns= {'description': 'count'})

items = items.sort_values(by=['count'], ascending = False)
items.head(6)

# description index as column
items.reset_index()
