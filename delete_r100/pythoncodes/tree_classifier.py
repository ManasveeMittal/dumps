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

import warnings 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class tree_classifier(object):


	def __init__(self, config, fp,data_path):
		self.config = config
		self.data = pd.read_csv(fp, sep = self.config['main']['data']['sep'])
		#self.data = self.data_.loc[np.logical_and(self.data_['pattern_1h01x'] == 0, self.data_['pattern_1h001'] == 0)]
	
		ignore_columns_ = [f.strip() for f in (self.config['main']['data']['ignore_features']+','+self.config['main']['data']['column_names']['y_column'].strip()).split(',')]
		ignore_columns = [ x for x in ignore_columns_ if x.strip()]
		print '[INFO] Ignore columns: ' + str(ignore_columns) + '\n'
		
		self.user_id = self.config['main']['data']['column_names']['user_id']
		self.y_label = self.config['main']['data']['column_names']['y_column']

		self.predictions_file = data_path +'/' + str(fp.split('/')[-1:][0]) + '_' + 'attrition_predictions.csv'
		self.model_file = data_path + '/attr_models/' + str(fp.split('/')[-1:][0]) + '_model.pkl'

		self.cut_off = float(self.config['main']['performance_params']['nTile_params']['cut_off'])
		self.entile = int(self.config['main']['performance_params']['nTile_params']['entile'])
		
		self.x_data = self.data.drop(ignore_columns, axis = 1, inplace=False)
		self.y_data = self.data[[self.config['main']['data']['column_names']['y_column']]]

		self.x_data = self.x_data.dropna()
		self.y_data = self.y_data.dropna()

		print '[INFO] Loaded ' + str(len(self.x_data)) + ' data points from: ' + fp + '\n'

#		self.x_data = self.x_data.apply(zscore)
#		self.x_data = self.x_data.apply(lambda x : x.fillna(x.mean()), axis = 1)
#		self.x_data = self.x_data.apply(pd.to_numeric, axis = 1)

#		nTiles = 5000
#		self.x_data = self.x_data.apply(lambda x : pd.cut(x.rank(), nTiles, labels = range(nTiles)), axis = 0)


	def make_splits(self):
		split_ratio = self.config['main']['test_set_fraction']
		self.x_train_, self.x_test_, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size = float(self.config['main']['test_set_fraction']), random_state=42)
		self.x_train = self.x_train_.drop(self.user_id, axis = 1, inplace = False)
		##########################################################################
		#### To run on entire Testset = (Entire data)
		#self.x_test_, self.y_test = self.x_data, self.y_data
		#self.x_test = self.x_test_.drop(self.user_id, axis = 1, inplace = False)
		#a#########################################################################
		#otherwise...
		self.x_test = self.x_test_.drop(self.user_id, axis = 1, inplace = False)
		
		##########################################################################
	
		#print 'test x, y length : '
		#print len(self.x_test), len(self.y_test)

		#Maintain the user_id map as train_test_split call does not maintain the index Ids as such...
		#self.x_test_id_idx_map = pd.merge(self.x_test_, self.data, on = self.user_id)[[self.user_id]]
		#y_test_id_idx_map = pd.merge(self.x_test_, self.y_test, left_index=True, right_index=True)
		#self.y_test_id_idx_map = pd.merge(y_test_id_idx_map, self.x_test_id_idx_map, on = self.user_id)[[self.user_id, self.y_label]]

		#print '[INFO] Train_set_features: ' + str(self.x_train.columns.values.tolist()) + '\n'

	def train_model(self):
		raise NotImplementedError("Please Implement this method")
	
	def test_model(self):	
		raise NotImplementedError("Please Implement this method")

	'''
	def dump_predictions(self,prediction_array):
		predicted_df = pd.DataFrame(prediction_array)
		predicted_df.columns = ['predicted_label']
		actual_df = self.y_test_id_idx_map
		actual_df.columns = [self.user_id, 'actual_label']

		final_df1 = pd.merge(self.x_test_id_idx_map, actual_df, on = self.user_id, how='inner')
		final_df2 = pd.merge(final_df1, predicted_df, left_index=True, right_index=True, how='inner')
		print len(self.x_test_id_idx_map,), len(actual_df), len(final_df1), len(final_df2)
		return final_df2[[self.user_id,'actual_label','predicted_label']]
	'''

	def test_entile_perfomance(self,probs):
		
		probs.columns= ['prob_0','prob_1']
		probs['predicted_class'] = np.where(probs['prob_1'] > self.cut_off,1,0)
		#print probs  
		
		
		actuals = pd.DataFrame(self.y_test)
		actuals.columns = ['actual_class']
		result = pd.concat([actuals,probs],axis = 1)
		result.sort_values(['prob_1'],ascending = False,inplace = True)	
		result['tps'] = np.where( (result['predicted_class'] == result['actual_class']) & (result['predicted_class']==1),1,0)
		result['entile_id'] = range(len(result))
		result['entile_id'] = np.floor((result['entile_id'] * self.entile) /len(result))
		entile_tps = result.groupby(['entile_id'])['tps'].sum().reset_index()
	        entile_tps.columns = ['entile_id','tp_sum']	
		#print result

		entile_lift_ = result[(result['predicted_class'] == 1) & (result['tps'] ==1)  ]
		#print entile_lift_['entile_id'].nunique
		entile_lift = entile_lift_.groupby(['entile_id'])['predicted_class'].sum().reset_index()	  
		entile_lift.columns = ['entile_id','entile_ones']	
				
		#print entile_lift

		train_df = pd.DataFrame(self.y_train)
		train_df.columns = ['lables']

		base_rate = float(len(train_df[(train_df['lables'] == 1)]))
		print '[INFO] base_rate: ' + str(base_rate)
		total_tp = len(result[(result['tps'] == 1)])
		
		result1 = pd.merge(result,entile_tps,left_on = 'entile_id', right_on = 'entile_id',how = 'outer')
		result2 = pd.merge(result1,entile_lift,left_on = 'entile_id', right_on = 'entile_id',how = 'outer')

		result2['percent_tp_entile'] = (result2['tp_sum'] * 100.0)/float(total_tp)

		result2['lift_each_tile'] = (self.entile * result2['entile_ones']) / float(base_rate)
		
		print result2[['entile_id','percent_tp_entile','lift_each_tile']].drop_duplicates()
		
		
			

	

	def print_feature_importances(self, f_imps):
		print '[INFO] Feature importances: '
		feature_imp = pd.DataFrame({'feature' : self.x_train.columns.values.tolist() , 'importance' : f_imps}).sort_values(['importance'], ascending = False)
		for idx, row in feature_imp.iterrows():
			print row['feature'] + ',' + str(row['importance'])

	def print_model_accuracy(self, preds):
		print '-'*75
		print 'Performance measures: '
		print 'Accuracy score: ' + str(pm.accuracy_score(self.y_test, preds))
		print 'Precision score: ' + str(pm.precision_score(self.y_test, preds))
		print 'Recall score: ' + str(pm.recall_score(self.y_test, preds))
		print 'F1 score: ' + str(pm.f1_score(self.y_test, preds))
		print 'MCC score: ' + str(pm.matthews_corrcoef(self.y_test, preds))
		print 'auROC curve: ' + str(pm.roc_auc_score(self.y_test, preds))
		print 'MCC score: ' + str(pm.matthews_corrcoef(self.y_test, preds))
		print 'Confusion matrix: \n' + str(pm.confusion_matrix(self.y_test, preds))
		print '-'*75


class dt_classifier(tree_classifier):

	def __init__(self, config, fp,data_path):
		tree_classifier.__init__(self, config, fp,data_path)

	def train_model(self):
		print '[INFO] Training Decision Tree with ' + str(len(self.x_train)) + ' data points.\n'

		self.dt = DecisionTreeClassifier(max_depth = int(self.config['main']['model_params']['decision_tree']['max_depth']), class_weight = 'balanced')
		#dtree = DecisionTreeClassifier(max_depth = int(self.config['main']['model_params']['decision_tree']['max_depth']))
#		print self.x_train.dtypes
#		dtree = self.dt.fit(self.x_train, self.y_train.values.ravel())
		dtree = self.dt.fit(self.x_train, self.y_train)

		print '[INFO] Finished training the Decision Tree.\n'
		joblib.dump(self.dt, self.model_file)		
		print '[INFO] Pickled model to file: ' + str(self.model_file)
                '''
		height = int(self.config['main']['model_params']['decision_tree']['max_depth'])
		for i in range(1,height):
		    print 'for attribute '+str(i)+' ' + str(self.dt.tree_.feature[i])+'error is ' + str(self.dt.tree_.best_error[i])			
		for i in range(1,height):
 		    print 'Entropy of all attributes at leve '+i+' is'+ str(self.dt.tree_.init_error[i])
                '''

		#self.print_feature_importances(self.dt.feature_importances_)

	def test_model(self):

		print '[PROGRESS] Testing DT model on ' + str(len(self.x_test)) + ' data points.\n'
		prediction_probs = pd.DataFrame(self.dt.predict_proba(self.x_test))#.merge(pd.DataFrame(self.y_test), left_index = True, right_index=True, how='inner')
		print(prediction_probs.columns)
		print(prediction_probs.head())
		prediction_probs.columns = ['0_prob', '1_prob']
		prediction_array = self.dt.predict(self.x_test)
		pred_df = pd.DataFrame(prediction_array, columns = ['predicted_label'])
		pred_df[self.user_id] = 0
		pred_df[self.y_label] = 0
		counter = 0
		for i, row in self.x_test.iterrows():
			uid = self.data.ix[i][self.user_id]
			y = self.y_test.ix[i][self.y_label]
			pred_df.ix[counter][self.user_id] = uid
			pred_df.ix[counter][self.y_label] = y
			counter += 1
		final_predictions = pd.concat([pred_df, prediction_probs], axis  = 1)
		final_predictions.to_csv(self.predictions_file,index = False)
		print '[PROGRESS] Predictions for test set written at location: ' + str(self.predictions_file)

		return final_predictions['predicted_label']

	def test_entile_perfomance(self):
	
		probs = pd.DataFrame(self.dt.predict_proba(self.x_test))
		super(dt_classifier,self).test_entile_perfomance(probs)

	def __exit__(self):
		del self

class rf_classifier(tree_classifier):

	def __init__(self, config, fp, data_path):
		tree_classifier.__init__(self, config, fp, data_path)

	def train_model(self):
		print '[INFO] Training Random Forest with ' + str(len(self.x_train)) + ' data points.\n'

		self.rf = RandomForestClassifier(n_estimators=int(self.config['main']['model_params']['random_forest']['n_estimators']), max_depth = int(self.config['main']['model_params']['random_forest']['max_depth']))
		self.rf.fit(self.x_train, self.y_train.values.ravel())

		print '[INFO] Finished training the RF.\n'

		joblib.dump(self.rf, self.model_file)		
		print '[INFO] Pickled model to file: ' + str(self.model_file)


	def test_model(self):
		print '[PROGRESS] Testing DT model on ' + str(len(self.x_test)) + ' data points.\n'
		prediction_probs = pd.DataFrame(self.rf.predict_proba(self.x_test))#.merge(pd.DataFrame(self.y_test), left_index = True, right_index=True, how='inner')
		print(prediction_probs.columns)
		print(prediction_probs.head())
		prediction_probs.columns = ['0_prob', '1_prob']
		prediction_array = self.rf.predict(self.x_test)
		pred_df = pd.DataFrame(prediction_array, columns = ['predicted_label'])
		pred_df[self.user_id] = 0
		pred_df[self.y_label] = 0
		counter = 0
		for i, row in self.x_test.iterrows():
			uid = self.data.ix[i][self.user_id]
			y = self.y_test.ix[i][self.y_label]
			pred_df.ix[counter][self.user_id] = uid
			pred_df.ix[counter][self.y_label] = y
			counter += 1
		final_predictions = pd.concat([pred_df, prediction_probs], axis  = 1)
		final_predictions.to_csv(self.predictions_file,index = False)

		print '[PROGRESS] Predictions for test set written at location: ' + str(self.predictions_file)

		return final_predictions['predicted_label']

	def test_entile_perfomance(self):

		probs = pd.DataFrame(self.rf.predict_proba(self.x_test))
		super(rf_classifier,self).test_entile_perfomance(probs)

	def __exit__(self):
		del self

