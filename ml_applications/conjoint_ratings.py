#!/usr/bin/env python3
'''
conjoint analysis (rating-based)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# data exploration
path_to_file = './data/survey.csv'
data = pd.read_csv(path_to_file)
data.head()

data.shape

# fit a linear model
X = data[['Career', 'Fitness', 'Humor', 'Religiosity', 'Interests']].copy()
y = data['Rating (1-5)']

X = pd.get_dummies(X, drop_first=True)

model = LinearRegression().fit(X, y)

df_coeff = pd.DataFrame(model.coef_, index=X.columns, columns=['coefficients'])

print(df_coeff.loc[df_coeff['coefficients'] > 0])
print(df_coeff.loc[df_coeff['coefficients'] < 0])

# attribute importance
range_coef = []
for i in data.columns[:-1]:
    levels_ref = pd.get_dummies(data[i]).columns  # same order as in X
    idx = [ i in s for s in X.columns.tolist() ]
    coef = np.r_[0, df_coeff.loc[idx, 'coefficients'].values]
    coef = pd.Series(coef, index=levels_ref)
    range_coef.append(max(coef.values) - min(coef.values))
    
    plt.plot(coef)
    plt.scatter(coef.index, coef.values)
    plt.xlabel(i)
    plt.ylabel('coefficients')
    plt.show()

importance = (100 * np.array(range_coef)/sum(range_coef)).round(2)

df = pd.DataFrame({'Attribute':data.columns[:-1], 'Range':range_coef, 'Importance': importance})

plt.bar(df['Attribute'], df['Importance'])
plt.xlabel('Attribute')
plt.ylabel('Importance')
plt.show()
