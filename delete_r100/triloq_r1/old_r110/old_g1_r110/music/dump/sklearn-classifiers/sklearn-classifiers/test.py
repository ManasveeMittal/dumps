import os
from glob import glob

run_config = {
            'model_dir': '/usr/local/triloq/platform/netbanking/attr_models/',
            'proddata_dir': '/usr/local/triloq/platform/netbanking/attr_proddata/',
            'out_dir' : '/usr/local/triloq/platform/netbanking/attr_scores/',
            'sep' : '|',
            'userid_col' : 'id',
            'ignore_cols' : 'casa_type'}

model_files = glob(run_config['model_dir'] + '*_model.pkl')
prod_files = glob(run_config['proddata_dir'] + '/*')
print(model_files)
print(prod_files)

data_files = []
for mf in model_files:
	f = mf.replace('TRAIN', 'PROD')
	ign, f = os.path.split(f.replace('_model.pkl', ''))
	for pf in prod_files:
		ign, bf = os.path.split(pf)
		if f == bf: data_files.append(pf)
    
for i in range(len(model_files)):
    print(model_files[i], data_files[i])

