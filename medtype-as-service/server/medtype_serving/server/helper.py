import argparse, pickle, logging, os, sys, time, uuid, warnings, itertools, pathlib
from collections import OrderedDict
from collections import defaultdict as ddict

import zmq
from termcolor import colored
from zmq.utils import jsonapi

__all__ = ['set_logger', 'get_args_parser', 'auto_bind', 'mergeList', 'clean_text', 'replace', 'check_file', 'TimeContext', 'CappedHistogram']


def set_logger(context, verbose=False):
	if os.name == 'nt':  # for Windows
		return NTLogger(context, verbose)

	logger = logging.getLogger(context)
	logger.setLevel(logging.DEBUG if verbose else logging.INFO)
	formatter = logging.Formatter(
		'%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
		'%m-%d %H:%M:%S')
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
	console_handler.setFormatter(formatter)
	logger.handlers = []
	logger.addHandler(console_handler)
	return logger


class NTLogger:
	def __init__(self, context, verbose):
		self.context = context
		self.verbose = verbose

	def info(self, msg, **kwargs):
		print('I:%s:%s' % (self.context, msg), flush=True)

	def debug(self, msg, **kwargs):
		if self.verbose:
			print('D:%s:%s' % (self.context, msg), flush=True)

	def error(self, msg, **kwargs):
		print('E:%s:%s' % (self.context, msg), flush=True)

	def warning(self, msg, **kwargs):
		print('W:%s:%s' % (self.context, msg), flush=True)


def check_max_seq_len(value):
	if value is None or value.lower() == 'none':
		return None
	try:
		ivalue = int(value)
		if ivalue <= 3:
			raise argparse.ArgumentTypeError("%s is an invalid int value must be >3 (account for maximum three special symbols in MedType model) or NONE" % value)
	except TypeError:
		raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
	return ivalue


def get_args_parser():
	from . import __version__

	parser = argparse.ArgumentParser(description='Start a MedTypeServer for serving')

	group1 = parser.add_argument_group('File Paths', 'config the path, checkpoint and filename of a pretrained/fine-tuned MedType model')
	group1.add_argument('--model_path', 		type=str, 	required=True, 			help='directory of a pretrained MedType model')
	group1.add_argument('--model_type', 		type=str, 	default='bert_combined', 	help='Model type. Options: [bert_combined (Bio arcticles/EHR), bert_plain (General)]')
	group1.add_argument('--entity_linker', 		type=str, 	default='scispacy',		help='entity linker to use over which MedType is employed. Options: \
														[scispacy, qumls, ctakes, metamap, metamaplite]')
	group1.add_argument('--tokenizer_model', 	type=str, 	default='bert-base-cased',	help='tokenizer to use')
	group1.add_argument('--context_len', 		type=int, 	default=120,			help='Number of tokens to consider in context for predicting mention semantic type')
	group1.add_argument('--model_batch_size', 	type=int, 	default=256, 			help='maximum number of mentions handled by MedType model')
	group1.add_argument('--threshold', 		type=float, 	default=0.5, 			help='Threshold used on logits from MedType')
	group1.add_argument('--type_remap_json', 	type=str, 	required=True, 			help='Json file containing semantic type remapping to coarse-grained types')
	group1.add_argument('--type2id_json', 		type=str, 	required=True, 			help='Json file containing semantic type to identifier mapping')
	group1.add_argument('--umls2type_file', 	type=str, 	required=True, 			help='Location where UMLS to semantic types mapping is stored')
	group1.add_argument('--dropout', 		type=float, 	default=0.1, 			help='Dropout in MedType Model.')

	# Entity Linkers Arguments
	group1.add_argument('--quickumls_path',		type=str, 	default=None, 			help='Location where QuickUMLS is installed')
	group1.add_argument('--metamap_path', 		type=str, 	default=None, 			help='Location where MetaMap executable is installed, e.g .../public_mm/bin/metamap18')
	group1.add_argument('--metamaplite_path', 	type=str, 	default=None, 			help='Location where MetaMapLite is installed, e.g .../public_mm_lite')
	group1.add_argument('--ctakes_host', 		type=str, 	default='localhost', 		help='IP at which cTakes server is running')
	group1.add_argument('--ctakes_port', 		type=int, 	default=9999, 			help='Port at which cTakes server is running')

	group2 = parser.add_argument_group('MedType Parameters', 'config how MedType model and pooling works')
	group2.add_argument('-max_seq_len', 		type=check_max_seq_len, default=25, help='maximum length of a sequence, longer sequence will be trimmed on the right side. '
												  'set it to NONE for dynamically using the longest sequence in a (mini)batch.')
	group2.add_argument('-cased_tokenization', 	dest='do_lower_case', 	action='store_false', default=True, help='Whether tokenizer should skip the default lowercasing and accent removal.'
															'Should be used for e.g. the multilingual cased pretrained MedType model.')
	
	group3 = parser.add_argument_group('Serving Configs', 'config how server utilizes GPU/CPU resources')
	group3.add_argument('--port', '-port_in', 	type=int, 		default=5555, 	help='server port for receiving data from client')
	group3.add_argument('--port_out', '-port_result',type=int, 		default=5556, 	help='server port for sending result to client')
	group3.add_argument('--http_port', 		type=int, 		default=None, 	help='server port for receiving HTTP requests')
	group3.add_argument('--http_max_connect', 	type=int, 		default=10, 	help='maximum number of concurrent HTTP connections')
	group3.add_argument('--enable_https', 		action='store_true', 			help='Enable serving HTTPs requests')
	group3.add_argument('--cors', 			type=str, 		default='*', 	help='setting "Access-Control-Allow-Origin" for HTTP requests')
	group3.add_argument('--num_worker', 		type=int, 		default=1, 	help='number of server instances')
	group3.add_argument('--max_batch_size', 	type=int, 		default=256, 	help='maximum number of sequences handled by each worker')
	group3.add_argument('--priority_batch_size', 	type=int, 		default=16, 	help='batch smaller than this size will be labeled as high priority,''and jumps forward in the job queue')
	group3.add_argument('--cpu', 			action='store_true', 	default=False, 	help='running on CPU (default on GPU)')
	group3.add_argument('--gpu_memory_fraction', 	type=float, 		default=0.5, 	help='determine the fraction of the overall amount of memory that each visible GPU should'
												     'be allocated per worker. Should be in range [0.0, 1.0]')
	group3.add_argument('--device_map', 		type=int, nargs='+', 	default=[], 	help='specify the list of GPU device ids that will be used (id starts from 0).' 
												     'If num_worker > len(device_map), then device will be reused; if num_worker < len(device_map), '
												     'then device_map[:num_worker] will be used')
	group3.add_argument('--prefetch_size', 		type=int, 		default=10, 	help='the number of batches to prefetch on each worker. When running on a CPU-only machine, this is set to 0 for comparability')
	parser.add_argument('--verbose', 		action='store_true', 	default=False, 	help='turn on tensorflow logging for debug')
	parser.add_argument('--version', 		action='version', 	version='%(prog)s ' + __version__)
	return parser


def auto_bind(socket):
	if os.name == 'nt':  # for Windows
		socket.bind_to_random_port('tcp://127.0.0.1')
	else:
		# Get the location for tmp file for sockets
		try:
			tmp_dir = '/tmp'
			if not os.path.exists(tmp_dir):
				raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
			tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
		except KeyError:
			tmp_dir = '*'

		socket.bind('ipc://{}'.format(tmp_dir))
	return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')


def get_run_args(parser_fn=get_args_parser, printed=True):
	args = parser_fn().parse_args()
	if printed:
		param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
		print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
	return args


def get_benchmark_parser():
	parser = get_args_parser()
	parser.description = 'Benchmark MedTypeServer locally'

	parser.set_defaults(num_client=1, client_batch_size=4096)

	group = parser.add_argument_group('Benchmark parameters', 'config the experiments of the benchmark')

	group.add_argument('-test_client_batch_size', type=int, nargs='*', default=[1, 16, 256, 4096])
	group.add_argument('-test_max_batch_size', type=int, nargs='*', default=[8, 32, 128, 512])
	group.add_argument('-test_max_seq_len', type=int, nargs='*', default=[32, 64, 128, 256])
	group.add_argument('-test_num_client', type=int, nargs='*', default=[1, 4, 16, 64])
	group.add_argument('-test_pooling_layer', type=int, nargs='*', default=[[-j] for j in range(1, 13)])

	group.add_argument('-wait_till_ready', type=int, default=30, help='seconds to wait until server is ready to serve')
	group.add_argument('-client_vocab_file', type=str, default='README.md', help='file path for building client vocabulary')
	group.add_argument('-num_repeat', type=int, default=10, help='number of repeats per experiment (must >2), as the first two results are omitted for warm-up effect')
	return parser


def get_shutdown_parser():
	parser = argparse.ArgumentParser()
	parser.description = 'Shutting down a MedTypeServer instance running on a specific port'

	parser.add_argument('-ip', type=str, default='localhost', help='the ip address that a MedTypeServer is running on')
	parser.add_argument('-port', '-port_in', '-port_data', type=int, required=True, help='the port that a MedTypeServer is running on')
	parser.add_argument('-timeout', type=int, default=5000, help='timeout (ms) for connecting to a server')
	return parser


class TimeContext:
	def __init__(self, msg):
		self._msg = msg

	def __enter__(self):
		self.start = time.perf_counter()
		print(self._msg, end=' ...\t', flush=True)

	def __exit__(self, typ, value, traceback):
		self.duration = time.perf_counter() - self.start
		print(colored('    [%3.3f secs]' % self.duration, 'green'), flush=True)

class CappedHistogram:
	"""Space capped dict with aggregate stat tracking.

	Evicts using LRU policy when at capacity; evicted elements are added to aggregate stats.
	Arguments:
	capacity -- the capacity limit of the dict
	"""
	def __init__(self, capacity):
		self.cache	= OrderedDict()
		self.capacity	= capacity
		self.base_bins	= 0
		self.base_count	= 0
		self.base_min	= float('inf')
		self.min_count	= 0
		self.base_max	= 0
		self.max_count	= 0

	def __getitem__(self, key):
		if key in self.cache:
			return self.cache[key]
		return 0

	def __setitem__(self, key, value):
		if key in self.cache:
			del self.cache[key]
		self.cache[key] = value
		if len(self.cache) > self.capacity:
			self._evict()

	def total_size(self):
		return self.base_bins + len(self.cache)

	def __len__(self):
		return len(self.cache)

	def values(self):
		return self.cache.values()

	def _evict(self):
		key,val = self.cache.popitem(False)
		self.base_bins += 1
		self.base_count += val
		if val < self.base_min:
			self.base_min	= val
			self.min_count	= 1
		elif val == self.base_min:
			self.min_count += 1
		if val > self.base_max:
			self.base_max	= val
			self.max_count	= 1
		elif val == self.base_max:
			self.max_count += 1

	def get_stat_map(self, name):
		if len(self.cache) == 0:
			return {}
		counts = self.cache.values()
		avg = (self.base_count + sum(counts)) / (self.base_bins + len(counts))
		min_, max_ = min(counts), max(counts)
		num_min, num_max = 0, 0
		if self.base_min <= min_:
			min_ = self.base_min
			num_min += self.min_count
		if self.base_min >= min_:
			num_min += sum(v == min_ for v in counts)

		if self.base_max >= max_:
			max_ = self.base_max
			num_max += self.max_count
		if self.base_max <= max_:
			num_max += sum(v == max_ for v in counts)

		return {
			'avg_%s' % name: avg,
			'min_%s' % name: min_,
			'max_%s' % name: max_,
			'num_min_%s' % name: num_min,
			'num_max_%s' % name: num_max,
		}

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

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

def check_file(filename):
	return pathlib.Path(filename).is_file()