#!/usr/bin/env python3
'''
A/B testing for three different marketing campaigns.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# data exploration
path_to_file = './data/marketing_campaign.csv'
df = pd.read_csv(path_to_file)
df.head()

df.shape

df['sales_in_thousands'].describe()

# sales per promotion strategy
sales_per_promo = df[['promotion','sales_in_thousands']].copy()
sales_per_promo = sales_per_promo.groupby(['promotion']).aggregate({'sales_in_thousands':'sum'})

sales_per_promo.idxmax()[0]

labels = [f'promotion {i}' for i in sales_per_promo.index.tolist()]
plt.pie(sales_per_promo['sales_in_thousands'], labels=labels, autopct='%2.2f %%')
plt.title("Sales by promotion")
plt.show()

# market size per promotion
df_market_size = df['market_size'].value_counts()
df_market_size

#df_market_size_per_promo = df.groupby(['promotion','market_size']).aggregate({'market_size':'count'})  # set promotion as index
df_market_size_per_promo = df.groupby(['promotion','market_size'], as_index=False).aggregate({'sales_in_thousands':'count'})
df_market_size_per_promo.rename(columns={'sales_in_thousands':'count'}, inplace=True)
df_market_size_per_promo

# alternatively:
df_market_size_per_promo = df.groupby(['promotion','market_size'], as_index=False)['sales_in_thousands'].count()

# alternatively:
df_market_size_per_promo = df.groupby(['promotion', 'market_size'])
df_market_size_per_promo = pd.DataFrame(df_market_size_per_promo['sales_in_thousands'].count()).reset_index()

# bar-plot
# using pd.pivot_table()
df_temp = df_market_size_per_promo.pivot_table(index='promotion', columns='market_size', values='count')
df_temp
# or using pd.crosstab()
df_temp = pd.crosstab(df['promotion'], df['market_size'])

df_temp.plot(kind='bar', stacked=False)
plt.title('market sizes by promotion')
plt.ylabel('count')
plt.show()

df_temp.plot(kind='bar', stacked=True)
plt.title('market sizes by promotion')
plt.ylabel('count')
plt.show()

# alternatively: using seaborn
'''
sbn.barplot(df_market_size_per_promo['promotion'], df_market_size_per_promo['sales_in_thousands'], df_market_size_per_promo['market_size'])
plt.title('market sizes by promotion')
plt.ylabel('count')
plt.show()

series1 = df_market_size_per_promo.loc[df_market_size_per_promo['market_size']== "Small"]
series2 = df_market_size_per_promo.loc[df_market_size_per_promo['market_size']== "Medium"]
series3 = df_market_size_per_promo.loc[df_market_size_per_promo['market_size']== "Large"]

sbn.barplot(series1['promotion'], series1['sales_in_thousands'], color='blue', label='Small')
sbn.barplot(series2['promotion'], series2['sales_in_thousands'], color='red', bottom=series1['sales_in_thousands'].tolist(), label='Medium')
sum_list = [a + b for a, b in zip(series1['sales_in_thousands'], series2['sales_in_thousands'])]
sbn.barplot(series3['promotion'], series3['sales_in_thousands'], color='green', bottom=sum_list, label = 'Large')

plt.title('market sizes by promotion')
plt.ylabel('count')
plt.legend()
plt.show()
'''

df_temp.T.idxmax()

# age of stores
df['store_age'].describe()

df['store_age'].value_counts()

fig, ax = plt.subplots(1,2)
ax[0].bar(df['store_age'].value_counts().index, df['store_age'].value_counts())
ax[0].set_xlabel('age of store')
ax[0].set_title('age distribution of stores')
ax[1].boxplot(df['store_age'])
plt.show()

pd.DataFrame(df['store_age'].value_counts()).idxmax()

# A/B testing
# scatter plot
plt.scatter(df['promotion'], df['sales_in_thousands'])
plt.xlabel('Promotion')
plt.ylabel('Sales in thousands')
#plt.xticks(ticks=[1,2,3], labels=[1,2,3])
plt.show()

# boxplot
sales1 = df.loc[df['promotion'] == 1, 'sales_in_thousands']
sales2 = df.loc[df['promotion'] == 2, 'sales_in_thousands']
sales3 = df.loc[df['promotion'] == 3, 'sales_in_thousands']

h = plt.boxplot([sales1, sales2, sales3], patch_artist=True, labels=[1,2,3], vert=False)
plt.setp(h['boxes'][0], color='red')
plt.setp(h['boxes'][2], color='green')
plt.ylabel('Promotion Strategy')
plt.xlabel('Sales in Thousands')
plt.title('Promotion Strategy vs. Sales')
plt.show()

# alternatively: using seaborn
'''
sbn.boxplot(y='promotion', x='sales_in_thousands', data=df, orient='h')
plt.xlabel("Promotion Strategy")
plt.ylabel("Sales in Thousands")
plt.title("Promotion Strategy vs. Sales")
plt.show()
'''

# statistical significance
means = df.groupby('promotion')['sales_in_thousands'].mean()
stds = df.groupby('promotion')["sales_in_thousands"].std()
ns = df.groupby('promotion')["sales_in_thousands"].count()

#promotion 1 vs. 2
t_1_vs_2 = (means.iloc[0] - means.iloc[1])/np.sqrt((stds.iloc[0]**2/ns.iloc[0]) + (stds.iloc[1]**2/ns.iloc[1]))
df_1_vs_2 = ((stds.iloc[0]**2/ns.iloc[0]) + (stds.iloc[1]**2/ns.iloc[1]))**2 / ( (stds.iloc[0]**2/ns.iloc[0])**2/(ns.iloc[0]-1) + (stds.iloc[1]**2/ns.iloc[1])**2/(ns.iloc[1]-1)) 
p_1_vs_2 = 2*(1 - stats.t.cdf(t_1_vs_2, df=df_1_vs_2))
print(f'p-value of promotion 1 vs. promotion 2: {p_1_vs_2}')

t, p = stats.ttest_ind(df.loc[df['promotion']==1, 'sales_in_thousands'].values,
                       df.loc[df['promotion']==2, 'sales_in_thousands'].values,
                       equal_var=False)

# promotion 1 vs. 3
t_1_vs_3 = (means.iloc[0] - means.iloc[2])/np.sqrt((stds.iloc[0]**2/ns.iloc[0]) + (stds.iloc[2]**2/ns.iloc[2]))
df_1_vs_3 = ((stds.iloc[0]**2/ns.iloc[0]) + (stds.iloc[2]**2/ns.iloc[2]))**2 / ( (stds.iloc[0]**2/ns.iloc[0])**2/(ns.iloc[0]-1) + (stds.iloc[2]**2/ns.iloc[2])**2/(ns.iloc[2]-1)) 
p_1_vs_3 = 2*(1 - stats.t.cdf(t_1_vs_3, df=df_1_vs_3))
print(f'p-value of promotion 1 vs. promotion 3: {p_1_vs_3}')

t, p = stats.ttest_ind(df.loc[df['promotion']==1, 'sales_in_thousands'].values,
                       df.loc[df['promotion']==3, 'sales_in_thousands'].values,
                       equal_var=False)

# promotion 3 vs. 2
t_3_vs_2 = (means.iloc[2] - means.iloc[1])/np.sqrt((stds.iloc[2]**2/ns.iloc[2]) + (stds.iloc[1]**2/ns.iloc[1]))
df_3_vs_2 = ((stds.iloc[2]**2/ns.iloc[2]) + (stds.iloc[1]**2/ns.iloc[1]))**2 / ( (stds.iloc[2]**2/ns.iloc[2])**2/(ns.iloc[2]-1) + (stds.iloc[1]**2/ns.iloc[1])**2/(ns.iloc[1]-1)) 
p_3_vs_2 = 2*(1 - stats.t.cdf(t_3_vs_2, df=df_3_vs_2))
print(f'p-value of promotion 3 vs. promotion 2: {p_3_vs_2}')

t, p = stats.ttest_ind(df.loc[df['promotion']==2, 'sales_in_thousands'],
                       df.loc[df['promotion']==3, 'sales_in_thousands'],
                       equal_var=False)

# ANOVA
anova = stats.f_oneway(df.loc[df['promotion']==1, 'sales_in_thousands'],
                       df.loc[df['promotion']==2, 'sales_in_thousands'],
                       df.loc[df['promotion']==3, 'sales_in_thousands'])
anova
anova.pvalue

# alternatively: using statsmodels
model = ols('sales_in_thousands ~ C(promotion)', data=df).fit()
table = sm.stats.anova_lm(model)
table

# Tukey's pairwise test
print(pairwise_tukeyhsd(df['sales_in_thousands'], df['promotion']))
