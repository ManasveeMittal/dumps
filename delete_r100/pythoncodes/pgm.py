%matplotlib inline

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


fp = '/media/triloq/tes3/nb/pgm/pgm_fs.tsv'

data_ = pd.read_csv(fp, sep = '|')

print(data.shape)
f1 = data[data['y'] <= 890167]
print(f1.shape)
f2 = f1[f1['avg_dr'] <= 1788902]
print(f2.shape)
data_ = f2


x_data = data['avg_dr'].reshape(-1,1)
y_data = data['y']
split_ratio = 0.4

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = split_ratio)


from sklearn import linear_model
reg = linear_model.LinearRegression()
model = reg.fit(x_train, y_train)


train_preds = reg.predict(x_train)
test_preds = reg.predict(x_test)


    

print(pm.r2_score(y_train, train_preds))
print(pm.r2_score(y_test, test_preds))


#data.boxplot(column = 'avg_dr')
#data.boxplot(column = 'y')
import matplotlib
matplotlib.pyplot.scatter(data['avg_dr'], data['y'])



import numpy as np

def extreme_vals(x) :
    print 'percentile ' + str(0.0) + ' : ' + str(round(x.min(),2))

    for t in np.concatenate([np.arange(0.01,0.1,0.01), np.arange(0.9,1.0,0.01)]):
        print 'percentile ' + str(t) + ' : ' + str(round(x.quantile(q=t),2))

    print 'percentile ' + str(1.0) + ' : ' + str(round(x.max(),2))
    
   
extreme_vals(data['avg_dr'])
#extreme_vals(data['y'])


t1 = np.arange(0.0,0.1,0.01) 
t2 = np.arange(0.9,1.0,0.01)
print(t1.shape)
print(t2.shape)

np.concatenate([t1 , t2])

print(data.shape)
f1 = data[data['y'] <= 890167]
print(f1.shape)
f2 = f1[f1['avg_dr'] <= 1788902]
print(f2.shape)


def r2(x,y):
    return [(x-y)].sd()

x = np.array([1,2,3])
y = np.array([6,7,8])
r2()