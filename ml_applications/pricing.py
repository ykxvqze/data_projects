#!/usr/bin/env python3
'''
pricing optimization
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

#
# data exploration
#

path_to_file = './data/wtp.txt'
df = pd.read_csv(path_to_file)
df.head()
df.shape

df['wtp'].count()
df.describe()

plt.hist(df['wtp'], bins=8, density=True)
plt.xlabel('wtp')
plt.ylabel("density")
plt.grid()
plt.show()

#
# fit a normal distribution
#

# method of moments
mean = df['wtp'].mean()
std = df['wtp'].std()
x = np.linspace(df['wtp'].min(), df['wtp'].max(), 100)
y = stats.norm.pdf(x=x, loc=mean, scale=std)

plt.hist(df['wtp'], density=True)
plt.plot(x, y, lw=2, label='theoretical')
plt.xlabel('wtp')
plt.ylabel('density')
plt.title('Normal distribution of wtp by method of moments')
plt.grid()
plt.legend()
plt.show()

# method of maximum-likelihood
mean, std = stats.distributions.norm.fit(df['wtp'])
y = stats.norm.pdf(x, mean, std)

plt.hist(df['wtp'], density=True)
plt.plot(x, y, lw=2, label='theoretical')
plt.xlabel('wtp')
plt.ylabel('density')
plt.title('Normal distribution of wtp by method of maximum-likelihood')
plt.grid()
plt.legend()
plt.show()

# same as 'std' by maximum-likelihood
df['wtp'].std() * np.sqrt((df['wtp'].count() - 1)/(df['wtp'].count()))

#
# fit a logistic distribution
#

# method of moments
loc = df['wtp'].mean()
scale = np.sqrt(df['wtp'].var() * 3 / np.pi**2)
x = np.linspace(df['wtp'].min(), df['wtp'].max(), 100)
y = stats.logistic.pdf(x=x, loc=loc, scale=scale)

plt.hist(df['wtp'], density=True)
plt.plot(x, y, lw=2, label='theoretical')
plt.xlabel('wtp')
plt.ylabel('density')
plt.title('Logistic distribution of wtp by method of moments')
plt.grid()
plt.legend()
plt.show()

# method of maximum-likelihood
loc, scale = stats.distributions.logistic.fit(df['wtp'])
y = stats.logistic.pdf(x, loc, scale)

plt.hist(df['wtp'], density=True)
plt.plot(x, y, lw=2, label='theoretical')
plt.xlabel('wtp')
plt.ylabel('density')
plt.title('Logistic distribution of wtp by method of maximum-likelihood')
plt.grid()
plt.legend()
plt.show()

#
# demand curve estimation
#

def demand(price, loc, scale, market_size):
    return market_size * (1 - stats.logistic.cdf(price, loc=loc, scale=scale))

demand_values = demand(x, loc, scale, 10000)

plt.plot(x, demand_values, lw=2)
plt.xlabel('price')
plt.ylabel('demand')
plt.title('Demand as a function of price')
plt.show()

#
# price optimization
#

def profit(price, cost, loc, scale, market_size):
    return demand(price, loc, scale, market_size) * (price - cost) 

print(profit(30, 15, loc, scale, 10000))
print(profit(20, 15, loc, scale, 10000))
print(profit(18, 15, loc, scale, 10000))

def neg_profit(price, cost, loc, scale, market_size):
    return -1 * profit(price, cost, loc, scale, market_size)

optimized = minimize(fun=neg_profit, x0=20, args=(15, loc, scale, 10000), method='BFGS')

# alternatively:
# optimized = minimize(lambda x : -1 * profit(x, 15, loc, scale, 10000), x0=20, method='BFGS')

# optimal price
price_opt = optimized.x

# optimal profit
profit_opt = -1 * optimized.fun  # or profit(optimized.x, 15, loc, scale, 10000)[0]

profit_values = profit(x, 15, loc, scale, 10000)

plt.plot(x, profit_values, lw=2)
plt.axhline(y=profit_opt, c='r', linestyle='--')
plt.axvline(x=price_opt, c='r', linestyle='--')
plt.xlabel('price')
plt.ylabel('profit')
plt.show()

#
# vectorized solution
#

market_size = 10000
demand_values = market_size * (1 - stats.logistic.cdf(x=x, loc=loc, scale=scale))

cost = 15
profit_values = demand_values * (x - cost)

profit_opt = max(profit_values)
price_opt = x[np.argmax(profit_values)]

plt.plot(x, profit_values, lw=2)
plt.axhline(y=profit_opt, c='r', linestyle='--')
plt.axvline(x=price_opt, c='r', linestyle='--')
plt.xlabel('price')
plt.ylabel('profit')
plt.show()
