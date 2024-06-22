#!/usr/bin/env python3
'''
churn prediction
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

# data exploration
path_to_file = './data/churn.txt'
data = pd.read_csv(path_to_file, sep=' ')
data.head()
data.shape

rows, cols = np.where(data.isna())
data.loc[rows]

data.dropna(inplace=True)
data.shape

data.nunique()

# distribution by gender
values, counts = np.unique(data['gender'], return_counts=True)

plt.bar(values, counts)
plt.xlabel('gender')
plt.ylabel('number of observations')
plt.title('Distribution by gender')
plt.show()

# distribution by internet service
series = data['internet_service'].value_counts()

plt.bar(series.index, series)
plt.xlabel('internet_service')
plt.ylabel('number of observations')
plt.title('Distribution by internet service')
plt.show()

series.idxmax()

# distribution by payment method
series = data['payment_method'].value_counts()

plt.bar(series.index, series)
plt.xlabel('payment_method')
plt.ylabel('number of observations')
plt.title('Distribution by payment method')
plt.show()

series.idxmax()

# feature transformation
# binary and continuous features
data.nunique()

df_binary = data[['gender','senior_citizen','partner','dependents','phone_service','paperless_billing','churn']].copy()
df_binary.replace({'Yes':1, 'No':0}, inplace=True)
df_binary.replace({'Male':1, 'Female':0}, inplace=True)

df_continuous = data[['tenure','monthly_charges','total_charges']]
df_continuous = (df_continuous - df_continuous.mean())/df_continuous.std()

df_continuous.describe().loc[['mean','std']]
df_continuous.std()

df_tx = pd.concat([df_continuous, df_binary], axis=1)

# multi-level categorical features
df_categorical = data[['multiple_lines','internet_service','online_security','online_backup',
                        'device_protection','tech_support','streaming_tv','streaming_movies',
                        'contract','payment_method']]

df_tx = pd.concat([pd.get_dummies(df_categorical), df_tx], axis=1)

df_tx.shape

# train/test set
X = df_tx.drop(['churn'], axis=1)
y = df_tx['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# artificial neural network with Keras/Tensorflow
model = keras.Sequential([
keras.layers.Dense(16, kernel_initializer="uniform", activation='relu', input_dim=X_train.shape[1]),
keras.layers.Dense(8, kernel_initializer="uniform", activation='relu'),
keras.layers.Dense(1, kernel_initializer="uniform", activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

model.fit(X_train, y_train, validation_split = 0.2, epochs = 40, batch_size = 10, verbose = True)

error_apparent = 1 - model.evaluate(X_train, y_train)[1]  # loss and accuracy
error_test = 1 - model.evaluate(X_test, y_test)[1]

model.predict(X_test)

# logistic classifier
model = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
model.score(X_test, y_test)  # accuracy

confusion_matrix(y_true=y_test, y_pred=y_pred)
error_test = (y_test!=y_pred).mean()

# confusion matrix as heatmap
'''
matrix = pd.DataFrame(confusion_matrix(y_true=y_test, y_pred=y_pred))
plt.matshow(matrix)
#plt.imshow(matrix))
#plt.locator_params(axis='x', nbins=2)
#plt.locator_params(axis='y', nbins=2)
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()

print(f'There are {matrix[0][0]} true negatives')
print(f'There are {matrix[0][1]} false negatives')
print(f'There are {matrix[1][0]} false positive')
print(f'There are {matrix[1][1]} true positive')
'''

# ROC/AUC using sklearn
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred_prob[:,0], pos_label=0)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC={roc_auc.round(2)}')
plt.plot([0, 1], [0, 1], c='gray', lw=1, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.show()

# ROC/AUC (manual implementation, for comparison with above)
tpr, fpr = [], []
for threshold in np.linspace(0,1,1000):
    labels = (y_pred_prob[:,1] > threshold)
    specificity = ((labels==0) & (y_test==0)).sum()/(y_test==0).sum()
    fpr.append(1 - specificity)
    tpr.append(((labels==1) & (y_test==1)).sum()/(y_test==1).sum())

def auc(x,y):
    dx = np.diff(x)  # x,y -> arrays
    base1 = y[:-1]  # base 1 of trapezium
    base2 = y[1:]
    trapezium_areas = (base1 + base2)*dx/2
    return(trapezium_areas.sum())

fpr, tpr = np.array(fpr), np.array(tpr)
AUC = auc(fpr[::-1], tpr[::-1])

plt.plot(fpr, tpr, label=f'AUC={AUC.round(2)}')
plt.plot([0, 1], [0, 1], c='gray', lw=1, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc="lower right")
plt.show()
