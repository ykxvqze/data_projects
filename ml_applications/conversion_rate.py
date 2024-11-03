#!/usr/bin/env python3
'''
customer conversion rate
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#
# data exploration
#

path_to_file = './data/conversion.txt'
data = pd.read_csv(path_to_file)
data.head()
data.shape

# conversion rates
data['y'].replace({'yes':1, 'no':0}, inplace=True)

data['y'].sum()/data['y'].count() * 100

values, counts = np.unique(data['y'], return_counts=True)
dict(zip(values,counts))

#
# visualization
#

# by number of contacts
df = data.groupby(['campaign']).aggregate({'campaign': 'count', 'y': 'sum'})
df.rename(columns={'campaign': 'count', 'y': 'n_conversions'}, inplace=True)
df['conversion_rate'] = df['n_conversions']/df['count'] * 100

plt.plot(df.index[:10], df['conversion_rate'][:10], c='b')
plt.scatter(df.index[:10], df['conversion_rate'][:10], c='b')
plt.xlabel('number of contacts')
plt.ylabel('conversion rate (%)')
plt.show()

# by age
df = data.groupby(['age']).aggregate({'age': 'count', 'y': 'sum'})
df.rename(columns={'age': 'count', 'y': 'n_conversions'}, inplace=True)
df['conversion_rate'] = df['n_conversions']/df['count'] * 100

plt.plot(df.index, df['conversion_rate'], c='b')
plt.scatter(df.index, df['conversion_rate'], c='b')
plt.xlabel('age')
plt.ylabel('conversion rate (%)')
plt.show()

# by age group
data['age_group'] =  data['age'].apply(lambda x: '<=20' if x<=20            \
                                         else '(20-30]' if (x>20 and x<=30) \
                                         else '(30-40]' if (x>30 and x<=40) \
                                         else '(40-50]' if (x>40 and x<=50) \
                                         else '(50-60]' if (x>50 and x<=60) \
                                         else '(60-70]' if (x>60 and x<=70) \
                                         else '70+')

df = data.groupby(['age_group']).aggregate({'age_group': 'count', 'y': 'sum'})
df.rename(columns={'age_group': 'count', 'y': 'n_conversions'}, inplace=True)
df['conversion_rate'] = df['n_conversions']/df['count'] * 100

# re-order for bar plot
df = df.reindex(['<=20','(20-30]','(30-40]','(40-50]','(50-60]','(60-70]','70+'])

plt.bar(df.index, df['conversion_rate'])
plt.xlabel('age group')
plt.ylabel('conversion rate (%)')
plt.show()

# by marital status
df = data.groupby(['marital']).aggregate({'marital': 'count', 'y': 'sum'})
df.rename(columns={'marital': 'count', 'y': 'n_conversions'}, inplace=True)
df['conversion_rate'] = df['n_conversions']/df['count'] * 100

labels = [f'{i}, {j} %' for i, j in zip(df.index.values, df['conversion_rate'].round(2))]

plt.pie(df['conversion_rate'], labels=labels, autopct='%0.2f%%')
plt.title('conversion rate by marital status')
plt.show()

# by education
df = data.groupby(['education']).aggregate({'education': 'count', 'y': 'sum'})
df.rename(columns={'education': 'count', 'y': 'n_conversions'}, inplace=True)
df['conversion_rate'] = df['n_conversions']/df['count'] * 100

labels = [f'{i}, {j} %' for i, j in zip(df.index.values, df['conversion_rate'].round(2))]

plt.pie(df['conversion_rate'], labels=labels, autopct='%0.2f%%')
plt.title('conversion rate by education')
plt.show()

# by job
df = data.groupby(['job']).aggregate({'job': 'count', 'y': 'sum'})
df.rename(columns={'job': 'count', 'y': 'n_conversions'}, inplace=True)
df['conversion_rate'] = df['n_conversions']/df['count'] * 100

plt.barh(df.index, df['conversion_rate'])
plt.xlabel("conversion rate (%)")
plt.show()

# duration of last contact
h = plt.boxplot([data.loc[data['y']==0,'duration']/3600, data.loc[data['y']==1,'duration']/3600], labels=[0, 1], patch_artist=True, vert=False)
plt.xlabel('duration (in hours)')
plt.ylabel('conversion status')
plt.title('last contact duration')
plt.setp(h["boxes"][0], color="red")
plt.setp(h['boxes'][1], color="blue")
plt.show() 

#
# logistic regression
#

df_categorical = pd.get_dummies(data[['marital','job','housing']])
df = pd.concat([df_categorical, data[['age','campaign','previous']]], axis=1)

X = df.copy()
y = data['y']

model = LogisticRegression(solver='lbfgs').fit(X, y)
y_pred = model.predict(X)

model.score(X, y)  # accuracy
model.coef_
model.intercept_

df = pd.DataFrame(model.coef_[0], index=X.columns.tolist(), columns=['coefficients'])

df.loc[df['coefficients']>0]
df.loc[df['coefficients']<0]

df.sort_values(by=['coefficients'], ascending=True)
# dict(zip(X.columns, model.coef_[0]))
