from configobj import ConfigObj
import os
import sys
import pandas as pd
from sklearn.externals import joblib
import shutil
from scipy.stats import zscore

from glob import glob
import argparse

def predict_scores(model, to_predict_base_path, sep, userid_col, ignore_cols, out_dir):

	print '[INFO] Predicting scores for producton base: ' + str(to_predict_base_path)

	#D = pd.read_csv(to_predict_base_path, sep = sep).head(1005)
	D = pd.read_csv(to_predict_base_path, sep = sep)
	user_map = D[[userid_col]]
	D.drop([userid_col] + ignore_cols, axis = 1, inplace = True)
	#D['d_d_customer_age'] = 25
	
#	D = D.apply(zscore)
#	nTiles = 5000
#	D = D.apply(lambda x : pd.cut(x.rank(), nTiles, labels = range(nTiles)), axis = 0)
#	D = D.apply(lambda x : x.fillna(x.mean()), axis = 1)
#	D = D.apply(pd.to_numeric, axis = 1)
	D = D.dropna()
	print 'classes_ = ' + str(model.classes_)
	prediction_probs = pd.DataFrame(model.predict_proba(D), columns = ['0_proba', '1_proba'])
	prediction_class = pd.DataFrame(model.predict(D), columns = ['predicted_label'])

	predictions = pd.concat([prediction_probs, prediction_class], axis = 1)
	R = predictions.merge(user_map, left_index = True, right_index = True)
	out_file = to_predict_base_path.split('/')[-1:][0]
	R.to_csv(out_dir + '/' + out_file + '.csv',sep=sep,index = False)
	print '[INFO] Scores written to file: ' + str(out_dir + '/' + out_file + '.csv')
	del D
	del R
		

if __name__ == '__main__':

	'''
	run_config = {
			'model_dir': '/usr/local/triloq/platform/netbanking/attr_models/',
			'proddata_dir': '/usr/local/triloq/platform/netbanking/attr_proddata/',
			'out_dir' : '/usr/local/triloq/platform/netbanking/attr_scores/',
			'sep' : '|',
			'userid_col' : 'id',
			#'ignore_cols' : 'casa_type,pattern_1h000'}
			'ignore_cols' : 'casa_type'}
	'''
	usage = '{} [-h|--help]'.format(os.path.basename(sys.argv[0]))	
	parser = argparse.ArgumentParser(usage)
	parser.add_argument('-c', action='store', dest='config_file', type=str, help='config file')
	parser.add_argument('-d', action='store', dest='data_path', type=str, help='data path')
	
	args = parser.parse_args()
	config_file = args.config_file
	data_path = args.data_path
	config = ConfigObj(config_file)		
	
	model_files = glob(data_path +'/attr_models/' + '*_model.pkl')
	prod_files = glob(data_path  +'/attr_proddata' + '/*')
	out_dir = data_path + '/attr_scores/'	
	data_files = []
	for mf in model_files:
		mf1 = mf.replace('train', 'prod')
		f = mf1.replace('TRAIN', 'prod')
#		f = f.replace('3mon', '10014')
		
		ign, f = os.path.split(f.replace('_model.pkl', ''))
		for pf in prod_files:
			ign, bf = os.path.split(pf)
			print f, bf
			if f.lower() == bf.lower(): 
				data_files.append(pf)
			
	print model_files
	print prod_files
	print data_files
	if os.path.isdir(out_dir):
		shutil.rmtree(out_dir)
		print 'Scoring in dir: [' + out_dir + '] already exist!'
		print 'Provide a different out_dir or delete the previous results directory first.'
		print 'Exiting...'
		#sys.exit(0)

	os.makedirs(out_dir)

	sep = config['run_config']['sep']
	userid_col = config['run_config']['userid_col']
	ignore_cols = config['run_config']['ignore_cols'].split(',')

	for i in range(len(model_files)):
		print "model_files ====== " + str(len(model_files))
		print "i is " + str(i) 
		print "data files" + str(data_files)
		print '[INFO] Loading model: ' + str(model_files[i])
		model = joblib.load(model_files[i])
		predict_scores(model, data_files[i], sep, userid_col, ignore_cols, out_dir)
	

