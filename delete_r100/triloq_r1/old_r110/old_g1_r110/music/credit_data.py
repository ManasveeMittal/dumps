from __future__ import print_function
__doc__ = 'Run all Classification algorithm with parameter search'
print(__doc__)

# IMPORT UTILITY FUNCTIONS
import logging
import numpy as np
import pandas as pd
import csv
from pprint import pprint
import matplotlib.pyplot as plt
from pylab import savefig
import pickle
from  datetime import datetime
import sys
from sklearn.datasets import load_boston
# from configobj import ConfigObj
# from __future__ import print_function

# IMPORT STATISTICAL FUNCTIONS
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import sklearn.metrics as pm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



#CONSTANT PARAMETERS FOR FUNCTIONS 
code_file = sys.argv[0]
data_dir = '/home/triloq/Music/credit_data/'
log_file = data_dir+code_file[:-3]+'.log'
logging.basicConfig(filename=log_file,level= logging.DEBUG )
random_state = 9000



#VARIABLE PARAMETERS FOR FUNCTIONS 
print_to_log = True
fname_1 = 'german.data-numeric.txt'


#FUNCTIONS AND OBJECTS
runtime= str(datetime.now())[:-7]
def log(value,take_log=True):
    logging.debug(value)

def param_grid(classifier):
    if classifier == 'DecisionTreeClassifier':
        param_grid_ = { 'criterion' : ['entropy','gini'],
                        'splitter' : ['best', 'random'],
                        'max_depth': [5,10,20]
                        # ,
                        # 'class_weight'  : ['balanced'],
                        # 'random_state' : [random_state]
                    }

    return param_grid_

#--------------------------------------------------------------
#IMPORT DATA
infile_1 = data_dir+fname_1
data = pd.read_csv(infile_1, delim_whitespace=True,low_memory=False,header=None)


#DATA OVERVIEW
def overview(data,first_n_cols = None,log_file=False):
    """ save/display data summary  """

    attributes = data.columns
    if first_n_cols != None:
        data = data[attributes[:first_n_cols]]
    log(value='\n shape '+ str(data.shape))
    log(value='\n first_n_rows '+ str(data.head()))
    log(value='\n attributes ' + str(attributes))
    log(value='\n data_description ' + str(data.describe()))
    log(value='\n data_information ' + str(data.info))
    log(value='\n overview end')

#DATA PREPROCESS : #TRAIN TEST SPLIT  
#COMMMETS: null value drop criteria not specified
id_col = None
y_label = 24
x_data = data.drop([y_label], axis = 1, inplace=False)
y_data = data[y_label]
split_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = split_ratio, random_state=random_state)

#MODEL FIT AND SCORE
classifiers = [
    DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, class_weight = 'balanced', random_state=random_state),
    RandomForestClassifier(criterion = 'entropy', max_depth = 5, class_weight = 'balanced', random_state=random_state),
    LogisticRegression(),
    GaussianNB(),
    SVC(kernel="linear", C=0.025)
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
]

def fit_predict_score(clf = None, display=True, save_file=None, persist=False, description='Not provided'):
    """take classifier and return metrics"""
    print(clf)
    model = clf.fit(x_train, y_train)
    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)


    if display==True:
        accuracy_scores = [round(pm.accuracy_score(    y_train, train_preds),2), round(pm.accuracy_score(    y_test, test_preds),2) ]
        precision_scores = [round(pm.precision_score(   y_train, train_preds),2), round(pm.precision_score(   y_test, test_preds),2) ]
        recall_scores = [round(pm.recall_score(      y_train, train_preds),2), round(pm.recall_score(      y_test, test_preds),2) ]
        f1_scores = [round(pm.f1_score(          y_train, train_preds),2), round(pm.f1_score(          y_test, test_preds),2) ]
        matthews_corrcoefs = [round(pm.matthews_corrcoef( y_train, train_preds),2), round(pm.matthews_corrcoef( y_test, test_preds),2) ]
        # roc_auc_scores = [round(pm.roc_auc_score(     y_train, train_preds),2), round(pm.roc_auc_score(     y_test, test_preds),2) ]
        matthews_corrcoefs = [round(pm.matthews_corrcoef( y_train, train_preds),2)      , round(pm.matthews_corrcoef( y_test, test_preds),2) ]

        print('-'*75)
        print('date            >' + runtime)
        print('Performance measures: ')
        print('Accuracy        > ' + 'train: ' + str(accuracy_scores[0]    ) + ' test: ' + str(accuracy_scores[1]    ))
        print('Precision score > ' + 'train: ' + str(precision_scores[0]   ) + ' test: ' + str(precision_scores[1]   ))
        print('Recall score    > ' + 'train: ' + str(recall_scores[0]      ) + ' test: ' + str(recall_scores[1]      ))
        print('F1 score        > ' + 'train: ' + str(f1_scores[0]          ) + ' test: ' + str(f1_scores[1]          ))
        print('MCC score       > ' + 'train: ' + str(matthews_corrcoefs[0] ) + ' test: ' + str(matthews_corrcoefs[1] ))
        # print('auROC curve     > ' + 'train: ' + str(roc_auc_scores[0]     ) + ' test: ' + str(roc_auc_scores[1]     ))
        print('MCC score       > ' + 'train: ' + str(matthews_corrcoefs[0] ) + ' test: ' + str(matthews_corrcoefs[1] ))
        print('train Confusion matrix: \n' + str(pm.confusion_matrix(y_train, train_preds)))
        print('\n')
        print('test Confusion matrix: \n' + str(pm.confusion_matrix(y_test, test_preds)))
        print('-'*75)

    if save_file != None:
        s = open(save_file,'a+')
        s.write('-'*75)
        s.write('\n')
        s.write('date            >' + runtime)
        s.write('\n')
        s.write(description)  
        s.write('\n')
        s.write('Performance measures: ')
        s.write('\n')
        s.write('Accuracy        > ' + 'train: ' + str(accuracy_scores[0]    ) + ' test: ' + str(accuracy_scores[1]    ))
        s.write('\n')
        s.write('Precision score > ' + 'train: ' + str(precision_scores[0]   ) + ' test: ' + str(precision_scores[1]   ))
        s.write('\n')
        s.write('Recall score    > ' + 'train: ' + str(recall_scores[0]      ) + ' test: ' + str(recall_scores[1]      ))
        s.write('\n')
        s.write('F1 score        > ' + 'train: ' + str(f1_scores[0]          ) + ' test: ' + str(f1_scores[1]          ))
        s.write('\n')
        s.write('MCC score       > ' + 'train: ' + str(matthews_corrcoefs[0] ) + ' test: ' + str(matthews_corrcoefs[1] ))
        s.write('\n')
        # s.write('auROC curve     > ' + 'train: ' + str(roc_auc_scores[0]     ) + ' test: ' + str(roc_auc_scores[1]     ))
        s.write('\n')
        s.write('MCC score       > ' + 'train: ' + str(matthews_corrcoefs[0] ) + ' test: ' + str(matthews_corrcoefs[1] ))
        s.write('\n')
        s.write('train Confusion matrix: \n' + str(pm.confusion_matrix(y_train, train_preds)))
        s.write('\n')
        s.write('test Confusion matrix: \n' + str(pm.confusion_matrix(y_test, test_preds)))
        s.write('-'*75)

    model_name = infile_1+'_'+str(clf).split('(')[0]+'.pkl'
    pickle.dump(model,open(model_name,'wb'))


# fit_predict_score(clf = classifiers[0], display=True, save_file=infile_1+'_metrics', persist=True, description=str(classifiers[0]))

# overview(data)
for clf in classifiers:
    fit_predict_score(clf = clf, display=True, save_file=infile_1+'_metrics', persist=False, description=str(clf))


