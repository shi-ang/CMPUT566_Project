# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:54:02 2019

@author: Shiang Qi
Estimating the generalisation performance of the algorithms
"""

import pickle
import numpy as np
import argparse
import os
from collections import Counter
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def get_top_n_features(data, label, num_runs = 1000, remaining_features_num = 10):
    idx_matrx = np.zeros([num_runs, remaining_features_num])
    for i in range(num_runs):
        clf = linear_model.SGDClassifier(penalty = 'elasticnet', alpha = 0.0001, max_iter = 1000, tol = 1e-6, shuffle = True, random_state = i)
        clf.fit(data, label)
        weights = clf.coef_
    

        weights_abs = np.abs(weights)
        top_feature_idx=weights_abs.squeeze().argsort()[::-1][0:remaining_features_num]
        idx_matrx[i, :] = top_feature_idx
    
    dic = Counter(idx_matrx.reshape(-1, 1).squeeze())
    top_features = dic.most_common(remaining_features_num)
    features_idx = np.zeros(remaining_features_num)
    for i in range(len(top_features)):
        features_idx[i] = top_features[i][0]
    return features_idx

def shuffled(a, b, seed = None):
    assert len(a) == len(b)
    if seed != None:
        np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':

    pickle_file = 'SH.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)  
        data = pickle_data['dataset']
        label = pickle_data['labels']
        del pickle_data 

    print('Data and modules loaded.')
    
    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--numruns_algs', type=int, default=1,
                        help='Specify the number of runs for algorithm selection')
    parser.add_argument('--numruns_features', type=int, default=1000,
                        help='Specify the number of runs for feature selection')
    parser.add_argument('--featuresleft', type=int, default=10,
                        help='Specify the number of features you want to keep')
    args = parser.parse_args()
    
    numruns_features = args.numruns_features
    numruns_algs = args.numruns_algs
    featuresleft = args.featuresleft
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(data, label, test_size = 0.3, random_state = 42)
    
    features_idx = get_top_n_features(Xtrain, ytrain, num_runs = numruns_features, 
                                      remaining_features_num = featuresleft).astype(np.int64)
    data_new = data[:, features_idx]
    data_new, label = shuffled(data_new, label, seed = 42)
    
    Xtrain = Xtrain[:, features_idx]
    Xtest = Xtest[:, features_idx]
    
    pickle_file = 'SH_selected_features.pickle'
    # determine whether the file is exist, if not, save date to pickle file
    if not os.path.isfile(pickle_file):    
        print('Saving data to pickle file...')
        try:
            with open('SH_selected_features.pickle', 'wb') as pfile:
                pickle.dump(
                        {
                            'train_data': Xtrain,
                            'train_label': ytrain,
                            'test_data': Xtest,
                            'test_label': ytest,
                            'whole_dataset': data_new,
                            'whole_label': label
                        },
                        pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    print('Data cached in pickle file.')
    
    # neigh = KNeighborsClassifier(n_neighbors = 3)
    # neigh.fit(data_new[0:40, :], label[0:40])
    # prediction_knn = neigh.predict(data_new[40:, :])
    # acc_knn = prediction_knn - label[40:]
    # clf = SVC(kernel = 'rbf')
    # clf.fit(data_new[0:40, :], label[0:40])
    # prediction_rbf = neigh.predict(data_new[40:, :])
    # acc_rbf = prediction_rbf - label[40:]


    
    # models = [SVC(kernel = 'rbf'), KNeighborsClassifier(), RandomForestClassifier(), DecisionTreeClassifier()]
    classalgs = {
        'SVM_rbf': SVC(kernel = 'rbf'),
        'SVM_linear': SVC(kernel = 'linear'),
        'LogisticRegression': LogisticRegression(penalty = 'l2', solver = 'liblinear'),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'GaussianNB': GaussianNB(),
        'RandomForestClassifier': RandomForestClassifier(n_estimators = 100, 
                                                          warm_start=True, 
                                                          max_features='sqrt',
                                                          max_depth=None, 
                                                          min_samples_split=2, 
                                                          min_samples_leaf=1, 
                                                          n_jobs=-1),
        # 'Neural Network': MLPClassifier(solver = 'adam', max_iter = 1000),
    }
    p_grid = {
        'SVM_rbf': {"C": [0.1, 1, 10], "gamma": [.01, .1]},
        'SVM_linear': {"C": [0.1, 1, 10], "gamma": [.01, .1]},
        'LogisticRegression': {"C": [0.1, 1, 10]},
        'KNeighborsClassifier': {"n_neighbors": np.arange(1, 10)},
        'Neural Network': {"hidden_layer_sizes":[5, 7, 9,
                                                 (5, 5), (5, 7), (5, 9),
                                                 (7, 5), (7, 7), (7, 9),
                                                 (9, 5), (9, 7), (9, 9),],
                            "activation": ['relu', 'logistic'],
        'RandomForest': {"n_estimators": [10, 100, 1000]}
        }
    }
    
    nested_scores = {}
    for algname in classalgs:
        nested_scores[algname] = np.zeros(numruns_algs)
    
    for run_times in range(numruns_algs):
        # data_new, label = shuffled(data_new, label)
        for algname, alg in classalgs.items():
            inner_cv = KFold(n_splits = (len(data_new) - 1), shuffle = True, 
                             random_state = run_times)
            outer_cv = KFold(n_splits = len(data_new), shuffle = True, 
                             random_state = run_times)
        
            clf = GridSearchCV(estimator = alg, 
                               param_grid = p_grid.get(algname, {}), 
                               cv = inner_cv, iid = False)
            
            clf.fit(data_new, label)
            nested_score = cross_val_score(clf, X = data_new, y = label, cv = outer_cv)
            nested_scores[algname][run_times] = nested_score.mean()
    
    for algname, acc in nested_scores.items():
        nested_scores[algname] = np.mean(acc)
    
    
        # example:
        # gs = GridSearchCV(estimator=pipe_svc, ... param_grid=param_grid, ... scoring='accuracy', ... cv=2)
        # scores = cross_val_score(gs, X_train, y_train, ... scoring='accuracy', cv=5)