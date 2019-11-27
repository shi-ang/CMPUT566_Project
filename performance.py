# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:56:40 2019

@author: Shiang Qi
Finalized the mode and compare the performance
"""
print(__doc__)
import generalization_ability
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    
    pickle_file = 'SH_selected_features.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)  
        Xtrain = pickle_data['train_data']
        Xtest = pickle_data['test_data']
        ytrain = pickle_data['train_label']
        ytest = pickle_data['test_label']
        data = pickle_data['whole_dataset']
        label = pickle_data['whole_label']
        del pickle_data 

    print('Data and modules loaded.')
    # label = label.astype(np.int32)
    # Xtrain, Xtest, ytrain, ytest = train_test_split(data, label, test_size = 0.8, random_state = 42)
    
    # p_grid = {"C": [0.01, 0.1, 1, 10], "gamma": [0.0001, 0.001, 0.01, 0.1]}
    # svm = SVC(kernel = 'linear', probability = True)
    classalgs = {
        # 'SVM_rbf': SVC(kernel = 'rbf', probability = True),
        'SVM_linear': SVC(kernel = 'linear', probability = True),
        # 'Loggs Regression': LogisticRegression(penalty = 'l2', 
        #                                         solver = 'liblinear'),
        # 'K Neighbors': KNeighborsClassifier(),
        # 'GaussianNB': GaussianNB(),
        # 'RandomForest': RandomForestClassifier(n_estimators = 100, 
                                                          # warm_start=True, 
                                                          # max_features='sqrt',
                                                          # max_depth=6, 
                                                          # min_samples_split=3, 
                                                          # min_samples_leaf=2, 
                                                          # n_jobs=-1, 
                                                          # verbose=0),
        # 'Neural Network': MLPClassifier(solver = 'adam', max_iter = 1000),
    }
    p_grid = {
        'SVM_rbf': {"C": [0.1, 1, 10], "gamma": [.01, .1]},
        'SVM_linear': {"C": [0.1, 1, 10], "gamma": [.01, .1]},
        'Loggs Regression': {"C": [0.1, 1, 10]},
        'K Neighbors': {"n_neighbors": np.arange(1, 10)},
        'Neural Network': {"hidden_layer_sizes":[
                                                 2, 3, 4, 5,
                                                 (2, 2), (2, 3), (2, 4), (2, 5),
                                                 (3, 2), (3, 3), (3, 4), (3, 5),
                                                 (4, 2), (4, 3), (4, 4), (4, 5),
                                                 (5, 2), (5, 3), (5, 4), (5, 5),],
                            "activation": ['relu', 'logistic']},
        'RandomForest': {"n_estimators": [10, 100, 1000]}
    }
    plt.figure(1)
    i = 0
    color = ['darkorange', 'green', 'red', 'blue', 'pink', 'cyan', 'purple', 'salmon']
    for alg_name, alg in classalgs.items():
        
        loo = LeaveOneOut()
        clf = GridSearchCV(estimator = alg, param_grid = p_grid.get(alg_name, {}), 
                           cv = loo, iid = False)
        clf.fit(Xtrain, ytrain)
        print(clf.best_params_)
        probas = clf.predict_proba(Xtest)[:, 1]
        acc = 1- np.sum(np.abs(clf.predict(Xtest) - ytest)) / len(ytest)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # for i in range(2):
        fpr, tpr, thresholds = roc_curve(ytest, probas, pos_label = 1)
        roc_auc = auc(fpr, tpr)
        
        
        linewidth = 2
        plt.plot(fpr, tpr, color = color[i], lw = linewidth,
                 label='%r ROC curve (area = %0.3f)' % (alg_name, roc_auc))
        i += 1
    
    plt.plot([0, 1], [0, 1], color = 'navy', lw = linewidth, linestyle = '--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc = "lower right")
    plt.show()