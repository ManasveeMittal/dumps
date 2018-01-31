import numpy as np
import pandas as pd
import csv
from configobj import ConfigObj
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import sklearn.metrics as pm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

#fp  = '/media/triloq/tes3/nb/result_delivery_201712/hab/hbl_traindata/feature_space_nbhbl_train_20171130'
#fp = '/media/triloq/tes3/nb/result_delivery_201712/NBATR_HDFC_IB_DEF_3MON10014/attr_traindata/feature_space_OTH_NBATR_HDFC_IB_DEF_3MON_TRAIN_20171130'
fp = '/media/triloq/tes3/nb/result_delivery_201712/NTILED500_NBATR_HDFC_IB_DEF_13014/attr_traindata/feature_space_ntiled_500_oth_nbatr_hdfc_ib_def_3mon_train_20171130'

data = pd.read_csv(fp, sep = '|')

#print(data.shape)
#print(data.columns)
#user_id = 'ucic_id'
#y_label = 'progress'

user_id = 'id'
y_label = 'pattern_1h000'
x_data = data.drop([user_id, y_label], axis = 1, inplace=False)
y_data = data[y_label]

#print(x_data.shape)
#print(y_data.shape)

x_data = x_data.dropna()
y_data = y_data.dropna()


pca = PCA(n_components=50)
pca.fit(x_data)
#print(pca.explained_variance_ratio_)
x_data_pc = pca.transform(x_data)


split_ratio = 0.4
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = split_ratio, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_data_pc, y_data, test_size = split_ratio, random_state=42)

#print(x_train_.shape)
#print(x_test_.shape)
#print(y_train.shape)
#print(y_test.shape)

#clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, class_weight = 'balanced', random_state=42)
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, max_features='auto', min_impurity_split=1e-07,random_state=42, class_weight='balanced')
#clf = LogisticRegressionCV()
#clf = LogisticRegression()

model = clf.fit(x_train, y_train)
train_preds = model.predict(x_train)
test_preds = model.predict(x_test)

print '-'*75
print 'Performance measures: '
print 'Accuracy        > ' + 'train: ' + str(round(pm.accuracy_score(    y_train, train_preds),2)) + ' test: ' + str(round(pm.accuracy_score(    y_test, test_preds),2))
print 'Precision score > ' + 'train: ' + str(round(pm.precision_score(   y_train, train_preds),2)) + ' test: ' + str(round(pm.precision_score(   y_test, test_preds),2))
print 'Recall score    > ' + 'train: ' + str(round(pm.recall_score(      y_train, train_preds),2)) + ' test: ' + str(round(pm.recall_score(      y_test, test_preds),2))
print 'F1 score        > ' + 'train: ' + str(round(pm.f1_score(          y_train, train_preds),2)) + ' test: ' + str(round(pm.f1_score(          y_test, test_preds),2))
print 'MCC score       > ' + 'train: ' + str(round(pm.matthews_corrcoef( y_train, train_preds),2)) + ' test: ' + str(round(pm.matthews_corrcoef( y_test, test_preds),2))
print 'auROC curve     > ' + 'train: ' + str(round(pm.roc_auc_score(     y_train, train_preds),2)) + ' test: ' + str(round(pm.roc_auc_score(     y_test, test_preds),2))
print 'MCC score       > ' + 'train: ' + str(round(pm.matthews_corrcoef( y_train, train_preds),2)) + ' test: ' + str(round(pm.matthews_corrcoef( y_test, test_preds),2))
print 'train Confusion matrix: \n' + str(pm.confusion_matrix(y_train, train_preds))
print('\n')
print 'test Confusion matrix: \n' + str(pm.confusion_matrix(y_test, test_preds))
print '-'*75


