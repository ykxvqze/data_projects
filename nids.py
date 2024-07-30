#!/usr/bin/env python3
'''
Multi-class classification via multiple one-class classification (OCC)
tasks using the Isolation Forest classifier. The dataset used is the
NSL-KDD dataset for building a network intrusion detection system.

DESCRIPTION:

The target class in any given one-versus-all (OVA) task is assigned a +1
label, while all other classes are assgined a -1 label. The optimal
(ROC) operating point criterion used is balanced accuracy, i.e.
1 - (fpr + (1-tpr))/2. The optimal threshold is found based on using 70%
of the training set. Class oversampling is applied via SMOTE before running
OVA tasks. Multiple one-versus-all (OVA) classifications are carried out in
the following sequence (most to least common): normal, DoS, Probe, R2L, U2R.
Instances that are not labeled by any of the one-class classifiers are
labeled by a nearest mean classifier.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

def relabel(y_train_orig, y_test_orig, class_label):
    '''
    y_train_orig, y_test_orig -> Series
    class_label               -> str
    '''
    y_train = y_train_orig.copy()
    y_test = y_test_orig.copy()
    
    y_train[y_train != class_label] = -1
    y_test[y_test != class_label] = -1
    
    y_train[y_train == class_label] = 1
    y_test[y_test == class_label] = 1
    
    y_train = pd.to_numeric(y_train)
    y_test = pd.to_numeric(y_test)
    
    return y_train, y_test

def split_data(X, y, label_train=1, ratio_train=0.7):
    size_train = int((y == label_train).sum()*ratio_train)
    sample = np.random.choice(X[y == label_train].index, size = size_train, replace = False)
    
    X_validate = X.loc[~ X.index.isin(sample)]
    y_validate = y.loc[~ y.index.isin(sample)]
    
    X_train = X.loc[X.index.isin(sample)]
    y_train = y.loc[y.index.isin(sample)]
    
    return X_train, y_train, X_validate, y_validate

def plot_roc(y_true, y_prob, label):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label = f'{label}: AUC={auroc.round(4)}')
    plt.plot([0, 1], [0, 1], c='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.axis('equal')
    plt.show()

def get_threshold(fpr, tpr, threshold_values):
    criterion_values = 1 - (fpr + (1-tpr))/2
    idx = np.argmax(criterion_values)
    threshold = threshold_values[idx]
    return threshold

def main():
    
    df_train = pd.read_csv('./data/KDDTrain+.txt', header = None)
    df_test = pd.read_csv('./data/KDDTest+.txt', header = None)
    
    print(df_train.head())
    
    features = ['duration','protocol_type','service','flag','src_bytes',
                'dst_bytes','land','wrong_fragment','urgent','hot',
                'num_failed_logins','logged_in','num_compromised',
                'root_shell','su_attempted','num_root','num_file_creations',
                'num_shells','num_access_files','num_outbound_cmds',
                'is_host_login','is_guest_login','count','srv_count',
                'serror_rate', 'srv_serror_rate','rerror_rate',
                'srv_rerror_rate','same_srv_rate','diff_srv_rate',
                'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
                'dst_host_same_srv_rate','dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
                'dst_host_serror_rate','dst_host_srv_serror_rate',
                'dst_host_rerror_rate','dst_host_srv_rerror_rate','attack',
                'score']
    
    df_train.columns = features
    df_test.columns = features
    
    df_train.drop('score', inplace=True, axis=1)
    df_test.drop('score', inplace=True, axis=1)
    
    print(df_train['attack'].value_counts())
    print(df_test['attack'].value_counts())
    
    # set metaclasses: normal, Probe, DoS, U2R, R2L
    Probe_items = ['satan','ipsweep','nmap','portsweep','mscan','saint']
    DoS_items = ['back','land','neptune','pod','smurf','teardrop','apache2',
                 'udpstorm','processtable','mailbomb']
    U2R_items = ['buffer_overflow','loadmodule','rootkit','perl',
                 'sqlattack','xterm','ps']
    R2L_items = ['guess_passwd','ftp_write','imap','phf','multihop',
                 'warezmaster','warezclient','spy','xlock','xsnoop',
                 'snmpguess','snmpgetattack','httptunnel','sendmail',
                 'named','worm']
    
    df_train['attack'].replace(Probe_items, 'Probe', inplace=True)
    df_train['attack'].replace(DoS_items, 'DoS', inplace=True)
    df_train['attack'].replace(U2R_items, 'U2R', inplace=True)
    df_train['attack'].replace(R2L_items, 'R2L', inplace=True)
    
    df_test['attack'].replace(Probe_items, 'Probe', inplace=True)
    df_test['attack'].replace(DoS_items, 'DoS', inplace=True)
    df_test['attack'].replace(U2R_items, 'U2R', inplace=True)
    df_test['attack'].replace(R2L_items, 'R2L', inplace=True)
    
    print(df_train['attack'].value_counts())
    print(df_test['attack'].value_counts())
    
    X_train = df_train.drop(['attack'], axis=1)
    y_train = df_train['attack']
    
    X_test = df_test.drop(['attack'], axis=1)
    y_test = df_test['attack']
    
    '''
    Combine the datasets since certain feature levels may be present
    in the test set, but missing in the training set. Then do
    one-hot-encoding and re-split again.
    '''
    
    X_train['train'] = 1
    X_test['train'] = 0
    df = pd.concat([X_train, X_test], axis=0)
    
    columns_to_encode = ['protocol_type','service','flag']
    
    df = pd.get_dummies(data=df, columns=columns_to_encode, drop_first=True)
    
    X_train = df[df['train']==1].copy()
    X_test = df[df['train']==0].copy()
    
    X_train.drop('train', inplace=True, axis=1)
    X_test.drop('train', inplace=True, axis=1)
    
    # normalize features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # dimensionality reduction (lda)
    lda = LDA().fit(X_train, y_train)
    X_train = lda.transform(X_train)
    X_test = lda.transform(X_test)
    
    for i in np.unique(y_train):
        plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], label=i)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='lower right')
    plt.show()
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    '''
    Apply class oversampling via SMOTE
    '''
    values, counts = np.unique(y_train, return_counts=True)
    counts[counts < max(counts)/2] = max(counts)/2
    strategy = dict(zip(values, counts))
    
    smote = SMOTE(sampling_strategy = strategy)
    
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    '''
    __ Done preprocessing __
    '''
    
    labels = ['normal','DoS','Probe','R2L','U2R']
    max_samples = [1000, 700, 500, 350, 35]
    
    predictions = np.tile(np.nan,(len(labels), y_test.shape[0]))
    
    for i, class_label in enumerate(labels):
        y_train_class, y_test_class = relabel(y_train, y_test, class_label)
        y_train_original = y_train_class.copy()
        X_train_class, y_train_class, X_validate_class, y_validate_class = split_data(X_train, y_train_class)
        
        isf = IsolationForest(n_estimators=500, max_samples=max_samples[i], n_jobs=-1)
        isf.fit(X_train_class)
        y_prob = isf.score_samples(X_validate_class)
        
        fpr, tpr, thresholds = roc_curve(y_validate_class, y_prob)
        plot_roc(y_true=y_validate_class, y_prob=y_prob, label=f'{class_label}-vs-all (validation set)')
        optimal_threshold = get_threshold(fpr, tpr, thresholds)
        y_pred = np.where(y_prob > optimal_threshold, 1, -1)
        print(classification_report(y_validate_class, y_pred))

        # test set
        '''
        Note: 'optimal_threshold' will be used in the final evaluation of the multiple OCC design.
        '''
        isf.fit(X_train[y_train_original==1])
        y_prob = isf.score_samples(X_test)
        fpr, tpr, thresholds = roc_curve(y_test_class, y_prob)
        plot_roc(y_true=y_test_class, y_prob=y_prob, label=f'{class_label}-vs-all (test set)')
        threshold = get_threshold(fpr, tpr, thresholds)
        y_pred = np.where(y_prob > threshold, 1, -1)
        print(classification_report(y_test_class, y_pred))
        
        y_pred = np.where(y_prob > optimal_threshold, 1, -1)
        predictions[i,:] = y_pred
    
    # nearest mean classifier (for instances rejected by all OCCs)
    nmc = NearestCentroid()
    nmc.fit(X_train, y_train)
    y_pred_nmc = nmc.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=y_pred_nmc))
    
    df = pd.DataFrame(np.vstack([predictions, y_pred_nmc]))
    
    # evaluation of final design
    y_pred = []
    cnt = 0
    
    for i in range(y_test.shape[0]):
        for j in range(len(labels)):
            if df.iloc[j, i] == 1:
                y_pred.append(labels[j])
                break
        if df.iloc[j, i] != 1:
            cnt += 1
            y_pred.append(y_pred_nmc[i])
    
    print(f'{cnt} out of {y_test.shape[0]} observations were classified by the nearest mean classifier.')
    print(classification_report(y_test, y_pred, target_names = labels))
    
    conf_mat = pd.DataFrame(confusion_matrix(y_true=y_test, y_pred=y_pred, labels=labels), columns=labels, index=labels)
    print(conf_mat)


if __name__ == '__main__':
    main()
