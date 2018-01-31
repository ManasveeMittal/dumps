
import os
import sys
from configobj import ConfigObj
import tree_classifier as classifiers
import shutil

import argparse

import multiprocessing
from concurrent import futures

import resource

config = ''
m_type = ''

class Logger(object):
	def __init__(self, log_file):
		self.terminal = sys.stdout
		self.log = open(log_file, "w")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		pass    




def run_base(model, fp, test_on_test, test_on_train, test_on_combined):
	print '#'*75
	print '[PROGRESS] Working on base: ' + str(fp)
	print '#'*75

	print '[PROGRESS] Making splits...'
	model.make_splits()
	print '[PROGRESS] Training model...'
	model.train_model()

	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
	if test_on_test == True:
		print '[PROGRESS] Testing model on Testset...'
		preds = model.test_model()
		print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		model.print_model_accuracy(preds)
	#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#	print '[PROGRESS] Testing N-Tile performance...'
#	model.test_entile_perfomance()
	print '#'*75


def work_with_one_file(fp, test_on_test, test_on_train, test_on_combined,data_path):
    print('===== processing: ' + fp)
    if os.path.isfile(fp):
        model = classifiers.dt_classifier(config, fp,data_path) if m_type == 'dt' else classifiers.rf_classifier(config, fp, data_path) if m_type == 'rf' else None
        if model == None:
            print '[ERROR] model type not recognized.'
            exit(0)
        base = os.path.basename(fp)
        run_base(model,fp, test_on_test, test_on_train, test_on_combined)
        del model

	return fp


if __name__ == '__main__':
	usage = '{} [-h|--help]'.format(os.path.basename(sys.argv[0]))
	parser = argparse.ArgumentParser(usage)
    #parser.add_argument('-c', action='store', dest='config_file', type=int, help='config file')
	parser.add_argument('-c', action='store', dest='config_file', help='config file')
	parser.add_argument('-d', action='store', dest='data_path', type=str, help='data path')
	parser.add_argument('-f', action='store', dest='input_file', help='Input file')
	parser.add_argument('-m', action='store_true', dest='first_file', help='Make directory')
	parser.add_argument('-p', action='store_true', dest='do_parallel', help='Parallelize')
	parser.add_argument('-ts', action='store_true', dest='test_on_test', help='Test on Testset')
	parser.add_argument('-tr', action='store_true', dest='test_on_train', help='Test on Trainset')
	parser.add_argument('-co', action='store_true', dest='test_on_combined', help='Test on Combined')
        
	args = parser.parse_args()
	if args.config_file == None:
		print(usage)
		sys.exit(1)

	config_file = args.config_file
	if not os.path.isfile(config_file):
		print'[ERROR] config file not found!'
		exit(0)

	config = ConfigObj(config_file)

	'''
	#Setup the logger instance
	if os.path.isdir(config['main']['runId']):
		if args.first_file:
			shutil.rmtree(config['main']['runId'])
		print 'Results for runId ' + ' [ ' + config['main']['runId'] + ' ] already exist!'
		print 'Provide a different runId or delete the previous results directory first.'
		print 'Exiting...'
		#sys.exit(0)
			
	if args.first_file:
		os.makedirs(config['main']['runId'])
	sys.stdout = Logger(config['main']['runId'] + '/' + config['main']['data']['log_filename'])
	'''
	m_type = config['main']['model'].strip().lower()

	if args.input_file != None:
		print('==== Processing: {}'.format(args.input_file))
		work_with_one_file(args.input_file, args.test_on_test, args.test_on_train, args.test_on_combined)
		sys.exit(0)

	data_path = args.data_path
	input_dir = data_path + '/attr_traindata/'
	print '[PROGRESS] Searching for input data in folder: ' + input_dir

	print('Number of CPUs: ' + str(multiprocessing.cpu_count()))
	file_queue = []
	for filename in os.listdir(input_dir):
		fp = input_dir + '/' + filename
		file_queue.append(fp)

	if args.do_parallel:
		print('Parallel processing the given input...')
		with futures.ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:
			res = executor.map(work_with_one_file, file_queue, args.test_on_test, args.test_on_train, args.test_on_combined)
			print(list(res))
	else:
		for fp in file_queue: work_with_one_file(fp, args.test_on_test, args.test_on_train, args.test_on_combined,data_path)    

