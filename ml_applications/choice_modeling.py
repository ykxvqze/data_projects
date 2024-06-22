#!/usr/bin/env python3
'''
conjoint analysis (choice modeling)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import mnlogit
from sklearn.linear_model import LogisticRegression

# data exploration
path_to_file = './data/tablets.txt'
data = pd.read_csv(path_to_file)
data.head()

data.shape  # (137*15*3, 10)

#
# Option 1: multinomial logit model
# no need for dummy vars for mnlogit(): does it internally;
# using dummy vars gives the same result
#
model = mnlogit('choice ~ 0 + brand + size + storage + ram + battery + price', data).fit()
model.summary()

model.params[['brand' in i for i in model.params.index.tolist()]]

# prediction
def predict_share(model, data):
    df = data.copy()
    df['alternative'] = model.predict(df)[1]
    df['alternative'] = np.exp(df['alternative'])
    df['alternative'] = df['alternative']/df['alternative'].sum()
    return df['alternative']

data_market = data.loc[[21,22,45,85], ['brand','size','storage','ram','battery','price']].copy()
data_market['predicted_share'] = predict_share(model, data_market)
data_market

data_new = data_market.copy()
data_new.loc[85,'ram'] = 'r4gb'
data_new['predicted_share'] = predict_share(model, data_new)
data_new

# willingness-to-pay
model.params

galaxy_to_ipad = (model.params.loc['brand[Galaxy]'] - model.params.loc['brand[iPad]'])/model.params.loc['price']
galaxy_to_ipad[0]

r1gb_to_r4gb = (0 - model.params.loc['ram[T.r4gb]'])/model.params.loc['price']
r1gb_to_r4gb[0]

sz7inch_to_sz9inch = (model.params.loc["size[T.sz7inch]"] - model.params.loc["size[T.sz9inch]"])/model.params.loc["price"]
sz7inch_to_sz9inch[0]

#
# Option 2: logistic regression (same result if all dummy vars are kept)
#

X = pd.get_dummies(data.drop(['choice','alternative_id','choiceset_id','consumer_id'], axis=1))
y = data['choice']

model = LogisticRegression(multi_class='multinomial', solver='newton-cg', fit_intercept=False).fit(X, y)

# parameter interpretation
model_parameters = pd.DataFrame(model.coef_[0], index=X.columns, columns=['coefficients'])
model_parameters

# prediction
data_market = X.loc[[21,22,45,85],:].copy()
prob = model.predict_proba(data_market)[:,1]
prob = np.exp(prob)/(np.exp(prob)).sum()  # softmax
data_market['predicted_share'] = prob
data_market

data_new = X.loc[[21,22,45,85],:].copy()
data_new.loc[85,'ram_r2gb'] = 0
data_new.loc[85,'ram_r4gb'] = 1
prob = model.predict_proba(data_new)[:,1]
prob = np.exp(prob)/(np.exp(prob)).sum()  # softmax
data_new['predicted_share'] = prob
data_new['predicted_share']

# willingness-to-pay
model_parameters

(model_parameters.loc['brand_Galaxy'] - model_parameters.loc['brand_iPad']) / model_parameters.loc['price']

(model_parameters.loc['ram_r1gb'] - model_parameters.loc['ram_r4gb']) / model_parameters.loc['price']

(model_parameters.loc['size_sz7inch'] - model_parameters.loc['size_sz9inch']) / model_parameters.loc['price']
