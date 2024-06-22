#!/usr/bin/env python3
'''
recommender systems
'''

# read-in data
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

path_to_file = './data/invoices.csv'
data = pd.read_csv(path_to_file)
data.head()
data.shape

# data preparation
data[data['quantity']<0].head(6)
df = data[data['quantity']>0]
df[df['customer_id'].isna()]
df['customer_id'].isna().sum()
df = df.dropna(subset=['customer_id'])  # or df[~df['customer_id'].isna()]
df.shape

# customer-item matrix
# using: pivot_table()
user_item_matrix = pd.pivot_table(df, index='customer_id', columns='stock_code', values='quantity', aggfunc='sum', fill_value=0)
user_item_matrix.shape

user_item_matrix = user_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
user_item_matrix.head(6)

df['stock_code'].nunique()
df['customer_id'].nunique()

# collaborative filtering
# user-based collaborative filtering
user_user_matrix = pd.DataFrame(cosine_similarity(user_item_matrix), columns=user_item_matrix.index, index=user_item_matrix.index)

user_user_matrix.head()

# making recommendations
customer_id = 13645
neighbors = user_user_matrix[customer_id].sort_values(ascending=False).index[1:10+1]
neighbors = neighbors.tolist()

ind = np.where(user_item_matrix.loc[customer_id,:]==1)
items_bought_by_customer = user_item_matrix.columns[ind].tolist()

ind = user_item_matrix.loc[neighbors].any(axis=0)
items_bought_by_neighbors = user_item_matrix.columns[ind].tolist()

items_to_recommend = list(set(items_bought_by_neighbors) - set(items_bought_by_customer))
# alternatively
items_to_recommend = [x for x in items_bought_by_neighbors if x not in items_bought_by_customer]

items_to_recommend_descriptions = df.loc[df['stock_code'].isin(items_to_recommend), ['stock_code','description']].drop_duplicates().set_index('stock_code')

items_to_recommend_descriptions.sort_values(by=['stock_code'], ascending=True)

# item-based collaborative filtering
item_item_matrix = pd.DataFrame(cosine_similarity(user_item_matrix.T), columns=user_item_matrix.columns, index=user_item_matrix.columns)
item_item_matrix.head()

stock_id = '71053'
similar_items = item_item_matrix[stock_id].sort_values(ascending=False).index[0:10+1]

items_to_recommend_descriptions = df.loc[df['stock_code'].isin(similar_items), ['stock_code','description']].drop_duplicates().set_index('stock_code')
items_to_recommend_descriptions.sort_values(by=['stock_code'], ascending=True)
