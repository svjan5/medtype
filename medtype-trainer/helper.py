import os, sys, numpy as np, random, argparse, codecs, time, json, csv, itertools
import logging, logging.config, pathlib, warnings, pickle
import torch, torch.nn as nn

from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict
from collections import defaultdict as ddict, Counter


def getChunks(inp_list, chunk_size):
	"""
	Splits inp_list into lists of size chunk_size
	Parameters
	----------
	inp_list:       List to be splittted
	chunk_size:     Size of each chunk required
	
	Returns
	-------
	chunks of the inp_list each of size chunk_size, last one can be smaller (leftout data)
	"""
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]


def make_dir(dirpath):
	"""
	Makes directory at the specified path if it does not exist

	Parameters
	----------
	dirpath:	Path to directory
	
	Returns
	-------
		
	"""
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

def checkFile(filename):
	"""
	Checks whether file exist at the given path or not
	Parameters
	----------
	filename:	File path
	
	Returns
	-------
		
	"""
	return pathlib.Path(filename).is_file()

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run
	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------
		
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object
	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open('{}/log_config.json'.format(config_dir)))
	config_dict['handlers']['file_handler']['filename'] = '{}/{}'.format(log_dir, name.replace('/', '-'))
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def str_proc(x):
	return str(x).strip().lower()

def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def dump_pickle(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))
	print('Pickled Dumped {}'.format(fname))

def load_pickle(fname):
	return pickle.load(open(fname, 'rb'))

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_gpu(batch, dev):
	batch_gpu = {}
	for key, val in batch.items():
		if   key.startswith('_'):	batch_gpu[key] = val
		elif type(val) == type({1:1}): 	batch_gpu[key] = {k: v.to(dev) for k, v in batch[key].items()}
		else: 				batch_gpu[key] = val.to(dev)
	return batch_gpu

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def clean_text(text):
	text = str(text.encode('ascii', 'replace').decode())
	text = text.replace('\n',' ')
	text = text.replace('|',' ')
	text = text.replace('\'',' ')
	return text

def replace(s, ch): 
	new_str = [] 
	l = len(s) 
	  
	for i in range(len(s)): 
		if (s[i] == ch and i != (l-1) and
		   i != 0 and s[i + 1] != ch and s[i-1] != ch): 
			new_str.append(s[i]) 
			  
		elif s[i] == ch: 
			if ((i != (l-1) and s[i + 1] == ch) and
			   (i != 0 and s[i-1] != ch)): 
				new_str.append(s[i]) 
				  
		else: 
			new_str.append(s[i]) 
		  
	return ("".join(i for i in new_str))