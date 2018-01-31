from glob import glob
import os
import sys
import argparse
# model_files = glob('/home/triloq/Music/' + '*.*')

model_files = glob('\\~/Music/' + '*.*')
print(model_files)

if __name__ == '__main__':
	usage = '{} [-h|--help]'.format(os.path.basename(sys.argv[0]))	
	parser = argparse.ArgumentParser(usage)
	parser.add_argument('-c', action='store', dest='config_file', type=str, help='config file')
	parser.add_argument('-d', action='store', dest='data_path', type=str, help='data path')
	args = parser.parse_args()
	config_file = args.config_file
	data_path = args.data_path
	print(args)