# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:56:40 2019

@author: Shiang Qi
Finalized the mode and compare the performance
Generate a stacking model
"""
print(__doc__)
import generalization_ability
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, KFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((NFOLDS, x_test.shape[0]))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

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


    classalgs = {
        'SVM_rbf': SVC(kernel = 'rbf', probability = True),
        'SVM_linear': SVC(kernel = 'linear', probability = True),
        'Loggs Regression': LogisticRegression(penalty = 'l2', 
                                                solver = 'liblinear'),
        'K Neighbors': KNeighborsClassifier(),
        'GaussianNB': GaussianNB(),
        # 'RandomForest': RandomForestClassifier(n_estimators = 1000, 
        #                                        warm_start=True, 
        #                                        max_features='sqrt',
        #                                        max_depth=None, 
        #                                        min_samples_split=2, 
        #                                        min_samples_leaf=1, 
        #                                        n_jobs=-1),
        # 'Neural Network': MLPClassifier(solver = 'adam', max_iter = 1000),
    }
    p_grid = {
        'SVM_rbf': {"C": [0.1, 1, 10], "gamma": [.01, .1]},
        'SVM_linear': {"C": [0.1, 1, 10], "gamma": [.01, .1]},
        'Loggs Regression': {"C": [0.1, 1, 10]},
        'K Neighbors': {"n_neighbors": np.arange(1, 10)},
        'Neural Network': {"hidden_layer_sizes":[5, 7, 9,
                                                 (5, 5), (5, 7), (5, 9),
                                                 (7, 5), (7, 7), (7, 9),
                                                 (9, 5), (9, 7), (9, 9),],
                            "activation": ['relu', 'logistic']},
        'RandomForest': {"n_estimators": [10, 100, 1000]}
    }
    plt.figure(1)
    i = 0
    color = ['darkorange', 'green', 'red', 'blue', 'pink', 'cyan', 'purple', 'salmon']
    linewidth = 2
    for alg_name, alg in classalgs.items():
        
        loo = LeaveOneOut()
        clf = GridSearchCV(estimator = alg, param_grid = p_grid.get(alg_name, {}), 
                           cv = loo, iid = False)
        clf.fit(Xtrain, ytrain)
        print(clf.best_params_)
        probas = clf.predict_proba(Xtest)[:, 1]
        acc = 1- np.sum(np.abs(clf.predict(Xtest) - ytest)) / len(ytest)
        print("%r model accuracy is: %0.3f" % (alg_name, acc))
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # for i in range(2):
        fpr, tpr, thresholds = roc_curve(ytest, probas, pos_label = 1)
        roc_auc = auc(fpr, tpr)
        
        
        plt.plot(fpr, tpr, color = color[i], lw = linewidth,
                 label='%r ROC curve (area = %0.3f)' % (alg_name, roc_auc))
        i += 1
    
    plt.plot([0, 1], [0, 1], color = 'navy', lw = linewidth, linestyle = '--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc = "lower right")
    plt.show()
    
    """
    Ensemble a stacking model:
        1st layer: KNN, Gaussian Naive Bayes, RandomForest
        2st layer: 
    """
    SEED = 0 # for reproducibility
    NFOLDS = 5 # set folds for out-of-fold prediction
    kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

    svmrbf = SVC(kernel = 'rbf', probability = True, gamma = 0.01, C = 1)
    svmlinear = SVC(kernel = 'linear', probability = True, gamma = 0.01, C = 10)
    lr = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 0.1)
    rf = RandomForestClassifier(n_estimators = 100, warm_start = False,
                                max_features='auto', max_depth=None, 
                                min_samples_split=2, min_samples_leaf = 1, 
                                n_jobs=-1)
    knn = KNeighborsClassifier(n_neighbors = 5)
    gnb = GaussianNB()
    nn = MLPClassifier(hidden_layer_sizes = (5, 7), solver = 'adam', max_iter = 1000, activation = 'relu')
    
    svmrbf_oof_train, svmrbf_oof_test = get_out_fold(svmrbf, Xtrain, ytrain, Xtest)
    svmlinear_oof_train, svmlinear_oof_test = get_out_fold(svmlinear, Xtrain, ytrain, Xtest)
    lr_oof_train, lr_oof_test = get_out_fold(lr, Xtrain, ytrain, Xtest)
    rf_oof_train, rf_oof_test = get_out_fold(rf, Xtrain, ytrain, Xtest)
    knn_oof_train, knn_oof_test = get_out_fold(knn, Xtrain, ytrain, Xtest)
    gnb_oof_train, gnb_oof_test = get_out_fold(gnb, Xtrain, ytrain, Xtest)
    nn_oof_train, nn_oof_test = get_out_fold(nn, Xtrain, ytrain, Xtest)
    
    x_train_2nd = np.concatenate((svmrbf_oof_train, svmlinear_oof_train, 
                                  lr_oof_train, rf_oof_train, knn_oof_train, 
                                  gnb_oof_train, nn_oof_train), axis = 1)
    x_test_2nd = np.concatenate((svmrbf_oof_test, svmlinear_oof_test, 
                                 lr_oof_test, rf_oof_test, knn_oof_test, 
                                 gnb_oof_test, nn_oof_test), axis = 1)
    

    clf = LogisticRegression(penalty = 'l2', C = 0.1, solver = 'liblinear')
    clf.fit(x_train_2nd, ytrain)
    
    # clf = RandomForestClassifier(n_estimators = 1000, warm_start=False, max_features='auto', max_depth= None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1)
    # clf.fit(x_train_2nd, ytrain)
    
    probas = clf.predict_proba(x_test_2nd)[:, 1]
    acc = 1 - np.sum(np.abs(clf.predict(x_test_2nd) - ytest)) / len(ytest)
    print("Stacking model accuracy: %0.3f" % (acc))
    # Compute ROC curve and ROC area for the stacking model
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, thresholds = roc_curve(ytest, probas, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    
    
    plt.figure()
    plt.plot(fpr, tpr, color = 'green', lw = linewidth, 
             label='%r ROC curve (area = %0.3f)' % ("Stacking Model", roc_auc))
    plt.plot([0, 1], [0, 1], color = 'navy', lw = linewidth, linestyle = '--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc = "lower right")
    plt.show()