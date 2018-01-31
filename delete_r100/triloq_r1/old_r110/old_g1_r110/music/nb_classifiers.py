#get_ipython().magic(u'matplotlib inline')

# import library functions
import numpy as np
import pandas as pd
import csv
impoprt sys
# from __future__ import print_function
from pprint import pprint
# from configobj import ConfigObj
import matplotlib.pyplot as plt
from pylab import savefig
import pickle
from  datetime import datetime

# import statistical functions
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



#functions
runtime= str(datetime.now())[:-7]
#import user defined functions

#import data
fp = '/home/triloq/Music/data/'
fname_1 = 'feature_space_ntiled_500_oth_nbatr_hdfc_ib_def_3mon_train_20171130'
infile_1 = fp+fname_1
data = pd.read_csv(infile_1, sep = '|',low_memory=False)

#view data 
total_cols = data.columns
max_cols = 10
print(len(total_cols),max_cols)
test_data = data[data.columns[:max_cols]]
# print(total_cols)
# print(data.head())
# print(data.info)

# data summary
data_description = data.describe()
# print(data.info())
# print(data_description)

# #test cols only
# # print(infile_1)
# fig = plt.figure()
# axes =fig.add_axes()
# axes.plot(data['r_overall_tottxncnt_cov'],range(0,1))
# # x=data[attr].plot.density()

# #visualizing density distributions
# n_test=1
# fig,ax=plt.subplots(n_test,1)
# for attr in total_cols[:n_test]:
    
#     print(attr)
#     x=data[attr].plot.density()
#     x.savefig('test1.png')



# columns = 3
# rows = 4
# fig,ax_array = plt.subplots(rows,columns)
# for i,ax_row in enumerate(ax_array):
#     for j,axes in enumerate(ax_row):
#         axes.set_title('{},{}'.format(i,j))
# plt.show()

# data['r_overall_txnvariety_sum'].plot.density()



# clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, class_weight = 'balanced', random_state=random_state)
# clf = RandomForestClassifier(criterion = 'entropy', max_depth = 10, class_weight = 'balanced', random_state=random_state)
# clf = LogisticRegression()
# clf = GaussianNB()
# clf = SVC(kernel="linear", C=0.025)
grid= GridSearchCV( DecisionTreeClassifier(), param_grid(classifier='DecisionTreeClassifier'), refit = True, verbose = 10)
 
model = grid.fit(x_train, y_train)
train_preds = model.predict(x_train)
test_preds = model.predict(x_test)

print('/n/n',\
	'grid.best_params_:,', grid.best_params_,'/n/n'\
,'grid.best_estimator_:', grid.best_estimator_, '/n/n'\
,'grid.grid_scores_:',grid.grid_scores_)


# steps = []
# steps.append(('standardize', StandardScaler()))
# steps.append(('classifier',LogisticRegression())) 
# clf = Pipeline(steps)



def metrics(display=True,save_file=None,description='Not provided'):

	if display==True:
		accuracy_scores = [round(pm.accuracy_score(    y_train, train_preds),2), round(pm.accuracy_score(    y_test, test_preds),2) ]
		precision_scores = [round(pm.precision_score(   y_train, train_preds),2), round(pm.precision_score(   y_test, test_preds),2) ]
		recall_scores = [round(pm.recall_score(      y_train, train_preds),2), round(pm.recall_score(      y_test, test_preds),2) ]
		f1_scores = [round(pm.f1_score(          y_train, train_preds),2), round(pm.f1_score(          y_test, test_preds),2) ]
		matthews_corrcoefs = [round(pm.matthews_corrcoef( y_train, train_preds),2), round(pm.matthews_corrcoef( y_test, test_preds),2) ]
		roc_auc_scores = [round(pm.roc_auc_score(     y_train, train_preds),2), round(pm.roc_auc_score(     y_test, test_preds),2) ]
		matthews_corrcoefs = [round(pm.matthews_corrcoef( y_train, train_preds),2)		, round(pm.matthews_corrcoef( y_test, test_preds),2) ]

		print('-'*75)
		print('date            >' + runtime)
		print('Performance measures: ')
		print('Accuracy        > ' + 'train: ' + str(accuracy_scores[0]    ) + ' test: ' + str(accuracy_scores[1]    ))
		print('Precision score > ' + 'train: ' + str(precision_scores[0]   ) + ' test: ' + str(precision_scores[1]   ))
		print('Recall score    > ' + 'train: ' + str(recall_scores[0]      ) + ' test: ' + str(recall_scores[1]      ))
		print('F1 score        > ' + 'train: ' + str(f1_scores[0]          ) + ' test: ' + str(f1_scores[1]          ))
		print('MCC score       > ' + 'train: ' + str(matthews_corrcoefs[0] ) + ' test: ' + str(matthews_corrcoefs[1] ))
		print('auROC curve     > ' + 'train: ' + str(roc_auc_scores[0]     ) + ' test: ' + str(roc_auc_scores[1]     ))
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
		s.write('auROC curve     > ' + 'train: ' + str(roc_auc_scores[0]     ) + ' test: ' + str(roc_auc_scores[1]     ))
		s.write('\n')
		s.write('MCC score       > ' + 'train: ' + str(matthews_corrcoefs[0] ) + ' test: ' + str(matthews_corrcoefs[1] ))
		s.write('\n')
		s.write('train Confusion matrix: \n' + str(pm.confusion_matrix(y_train, train_preds)))
		s.write('\n')
		s.write('test Confusion matrix: \n' + str(pm.confusion_matrix(y_test, test_preds)))
		s.write('-'*75)


# description="DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, class_weight = 'balanced', random_state=random_state)"
# # metrics(save_file=infile_1+'_metrics', description=description)
metrics()
model_name = infile_1+'_DecisionTreeClassifier.pkl'
# pickle.dump(model,open(model_name,'wb'))




