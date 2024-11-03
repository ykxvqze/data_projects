#!/usr/bin/env python3
'''
customer lifetime value
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data exploration
path_to_file = './data/transactions.txt'
data = pd.read_csv(path_to_file, sep='\t', header=None, names=['id','amount','date'])
data.head()
data.shape

'''
data.columns = ['id','amount','date']
data.index = data['id']
data.set_index('id')
'''

data.describe()

data['date'] = pd.to_datetime(data['date'])

data['date'].min()
data['date'].max()

data['year'] = data['date'].dt.year
data.head()

# RFM features
data['days_elapsed'] = (pd.to_datetime('2016-01-01') - data['date']).dt.days

data_2015 = data.groupby(['id']).aggregate({'days_elapsed':['min', 'max'], 'id':'count', 'amount':'mean'})  # both min and max
data_2015.columns = ['recency','first_purchase','frequency','average_amount']
data_2015.head()

# revenues
rev_2015 = data[['id','amount']].copy()
rev_2015.loc[data['year']!=2015, 'amount'] = 0

rev_2015 = rev_2015.groupby(['id']).aggregate({'amount':'sum'})
rev_2015.columns = ['rev_2015']

# past data
data_2014 = data.copy().loc[data['days_elapsed']>365]  # or data_2014 = data[data['date']<'2015-01-01']

data_2014 = data_2014.groupby(['id']).aggregate({'days_elapsed':['min', 'max'], 'id':'count', 'amount':'mean'})
data_2014.columns = ['recency','first_purchase','frequency','average_amount']
data_2014[['recency','first_purchase']] -= 365
data_2014.head()

# data merging
data_merged = pd.merge(data_2014, rev_2015, on='id', how='left')

# segmentation
data_2015['cluster'] = np.nan
data_2015.loc[data_2015['recency']>365*3, 'cluster'] = 'A'
data_2015.loc[(data_2015['recency']>365*2) & (data_2015['recency']<=365*3), 'cluster'] = 'B'
data_2015.loc[(data_2015['recency']>365) & (data_2015['recency']<=365*2), 'cluster'] = 'C'
data_2015.loc[(data_2015['recency']<365), 'cluster'] = 'D'
data_2015.loc[(data_2015['cluster'] == 'C') & (data_2015['first_purchase'] <= 365*2), 'cluster'] = 'E'
data_2015.loc[(data_2015['cluster'] == 'C') & (data_2015['average_amount'] < 100), 'cluster'] = 'F'
data_2015.loc[(data_2015['cluster'] == 'C') & (data_2015['average_amount'] >= 100), 'cluster'] = 'G'
data_2015.loc[(data_2015['cluster'] == 'D') & (data_2015['first_purchase'] <= 365), 'cluster'] = 'H'
data_2015.loc[(data_2015['cluster'] == 'D') & (data_2015['average_amount'] < 100), 'cluster'] = 'I'
data_2015.loc[(data_2015['cluster'] == 'D') & (data_2015['average_amount'] >= 100), 'cluster'] = 'J'

data_2015.head()

data_2015['cluster'].value_counts().sort_index()
data_2015.groupby(['cluster']).mean()

# fix the indexing innately by transforming the column to a categorical var
data_2015['cluster'] = pd.Categorical(data_2015['cluster'], ['A','B','G','F','E','J','I','H'])
data_2015.groupby(['cluster']).mean()

#data_2015['cluster'].value_counts().reindex(['A','B','G','F','E','J','I','H'])
#data_2015.groupby(['cluster']).mean().reindex(['A','B','G','F','E','J','I','H'])  # re-order segments

# same for data_2014
data_2014['cluster'] = np.nan
data_2014.loc[data_2014['recency']>365*3, 'cluster'] = 'A'
data_2014.loc[(data_2014['recency']>365*2) & (data_2014['recency']<=365*3), 'cluster'] = 'B'
data_2014.loc[(data_2014['recency']>365) & (data_2014['recency']<=365*2), 'cluster'] = 'C'
data_2014.loc[(data_2014['recency']<365), 'cluster'] = 'D'
data_2014.loc[(data_2014['cluster'] == 'C') & (data_2014['first_purchase'] <= 365*2), 'cluster'] = 'E'
data_2014.loc[(data_2014['cluster'] == 'C') & (data_2014['average_amount'] < 100), 'cluster'] = 'F'
data_2014.loc[(data_2014['cluster'] == 'C') & (data_2014['average_amount'] >= 100), 'cluster'] = 'G'
data_2014.loc[(data_2014['cluster'] == 'D') & (data_2014['first_purchase'] <= 365), 'cluster'] = 'H'
data_2014.loc[(data_2014['cluster'] == 'D') & (data_2014['average_amount'] < 100), 'cluster'] = 'I'
data_2014.loc[(data_2014['cluster'] == 'D') & (data_2014['average_amount'] >= 100), 'cluster'] = 'J'

data_2014.head()

data_2014['cluster'].value_counts().sort_index()
data_2014.groupby(['cluster']).mean()

# fix the indexing innately by transforming the column to a categorical var
data_2014['cluster'] = pd.Categorical(data_2014['cluster'], ['A','B','G','F','E','J','I','H'])
data_2014.groupby(['cluster']).mean()

#data_2014['cluster'].value_counts().reindex(['A','B','G','F','E','J','I','H'])
#data_2014.groupby(['cluster']).mean().reindex(['A','B','G','F','E','J','I','H'])

df = data_2014['cluster'].value_counts()
plt.pie(df.values, labels=df.index, autopct='%0.2f%%')
plt.title('Customer segments')
plt.show()

df.idxmax()  # largest segment

# revenue per cluster
df_rev = data_merged.groupby(data_2014['cluster']).aggregate({'rev_2015':'mean'})

df_rev = df_rev.sort_values(by='rev_2015', ascending=False)

plt.bar(df_rev.index, df_rev['rev_2015'])
plt.title('Average revenue per cluster')
plt.show()

df_rev['rev_2015'].idxmax()  # cluster with highest revenue

# rank of 'H'
(df_rev.index=='H').argmax() + 1

# Markov model
df = pd.merge(data_2014, data_2015, on='id', how='left')
df.head()

M = pd.crosstab(df['cluster_x'], df['cluster_y'])
M['H'] = 0  # force column 'H' to appear
M = M.divide(M.sum(axis=1), axis='index')

M.loc['A','A']
M.loc['H','E']

# evolution of customer segments
clusters = pd.DataFrame(np.nan, columns=range(2015,2025+1), index=['A','B','G','F','E','J','I','H'])
clusters[2015] = data_2015['cluster'].value_counts()

for year in range(2016,2025+1):  # or clusters.columns[1:]
    clusters[year] = clusters[year-1].dot(M)

clusters.round()

# doesn't force all xticks (years) to show
'''
plt.bar(clusters.columns, clusters.loc['A',:])
plt.title('Evolution of number of customers in cluster A')
plt.show()
'''

plt.bar(range(len(clusters.columns)), clusters.loc['A',:])
plt.title('Evolution of number of customers in cluster A')
plt.xticks(ticks=range(11), labels=clusters.columns)
plt.show()

plt.bar(range(len(clusters.columns)), clusters.loc['B',:])
plt.title('Evolution of number of customers in cluster B')
plt.xticks(ticks=range(11), labels=clusters.columns)
plt.show()

# yearly revenue per segment
# many zeros since those segments have recency > 1 year
rev_yearly = rev_2015.groupby(data_2015['cluster']).aggregate({'rev_2015':'mean'})

# avg yearly rev per segment * number of customers (this works since we have unique customers in each segment)
rev_per_cluster = clusters.multiply(rev_yearly['rev_2015'], axis='index')

rev_per_cluster.sum(axis=0).round()

plt.bar(rev_per_cluster.sum(axis=0).index, rev_per_cluster.sum(axis=0))
plt.title('Total revenue in each year')
plt.show()

rev_per_cluster.sum(axis=0).cumsum()

plt.bar(rev_per_cluster.sum(axis=0).cumsum().index, rev_per_cluster.sum(axis=0).cumsum())
plt.title('Cumulative revenue')
plt.show()

# accounting for interest rate
i = 0.1
P = 1 / ((1 + i)**np.arange(0,11))  # P = F/(1+i)^n

total_rev_yearly_disc = rev_per_cluster.sum(axis=0) * P

# alternatively:
#total_rev_yearly_disc = [x*(1/(1.1**i)) for i,x in enumerate(rev_per_cluster.sum(axis=0))]

#plt.bar(total_rev_yearly_disc.index, total_rev_yearly_disc)
plt.bar(range(11), total_rev_yearly_disc)
plt.xticks(ticks=range(11), labels=clusters.columns)
plt.title('Total revenue in each year')
plt.show()

total_cumul_disc = total_rev_yearly_disc.cumsum()

plt.bar(range(11), total_cumul_disc)
plt.xticks(ticks=range(11), labels=clusters.columns)
plt.title('Cumulative revenue')
plt.show()

# business worth (expected revenue over this period):
total_cumul_disc.iloc[-1] - total_cumul_disc.iloc[0]  # or total_rev_yearly_disc.iloc[1:].sum()
