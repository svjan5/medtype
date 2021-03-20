#!/usr/bin/env python

import multiprocessing, threading, os, random, sys, json, time, numpy as np, pickle

from itertools import chain
from datetime import datetime
from multiprocessing import Process
from multiprocessing.pool import Pool
from nltk.tokenize.punkt import PunktSentenceTokenizer
from collections import OrderedDict
from collections import defaultdict as ddict

import torch
import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from .helper import *
from .http import MedTypeHTTPProxy
from .zmq_decor import multi_socket

from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from .medtype.models import BertCombined, BertPlain
from .entity_linkers import ScispaCy, QUMLS, CTakes, Metamap, MetamapLite

__all__ = ['__version__', 'MedTypeServer']
__version__ = '1.0.0'

class ServerCmd:
	terminate	= b'TERMINATION'
	show_config	= b'SHOW_CONFIG'
	show_status	= b'SHOW_STATUS'
	new_job		= b'REGISTER'
	elink_out	= b'ELINKS'

	@staticmethod
	def is_valid(cmd):
		return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())


class MedTypeServer(threading.Thread):
	def __init__(self, args):
		super().__init__()
		self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

		self.model_path			= args.model_path 		# Location where BERT model is stored
		self.max_seq_len		= args.max_seq_len 		# Model related argument
		self.num_worker			= args.num_worker 		# Number of server workers
		self.max_batch_size		= args.max_batch_size 		# Maximum batch size
		self.num_concurrent_socket	= max(8, args.num_worker * 2)  	# optimize concurrency for multi-clients
		self.port			= args.port 			# Port of ventilator PULLING users queries
		self.args			= args
		self.status_args		= {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
		self.status_static = {
			'python_version'	: sys.version,
			'server_version'	: __version__,
			'pyzmq_version'		: zmq.pyzmq_version(),
			'zmq_version'		: zmq.zmq_version(),
			'server_start_time'	: str(datetime.now()),
		}
		self.processes = []
		self.logger.info('freeze, optimize and export graph, could take a while...')

		self.model_params	= self.load_model(args.model_path)
		self.type_remap		= json.load(open(args.type_remap_json))
		self.type2id		= json.load(open(args.type2id_json))
		self.umls2type		= pickle.load(open(args.umls2type_file, 'rb'))

		self.is_ready = threading.Event()

	def load_model(self, model_path):
		state 		= torch.load(model_path, map_location="cpu")
		state_dict	= state['state_dict']
		new_state_dict	= OrderedDict()

		for k, v in state_dict.items():
			if 'module' in k:
				k = k.replace('module.', '')
			new_state_dict[k] = v

		return new_state_dict

	def __enter__(self):
		self.start()
		self.is_ready.wait()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()

	def close(self):
		self.logger.info('shutting down...')
		self._send_close_signal()
		self.is_ready.clear()
		self.join()

	@zmqd.context()
	@zmqd.socket(zmq.PUSH)
	def _send_close_signal(self, _, frontend):
		frontend.connect('tcp://localhost:%d' % self.port)
		frontend.send_multipart([b'0', ServerCmd.terminate, b'0', b'0'])

	@staticmethod
	def shutdown(args):
		with zmq.Context() as ctx:
			ctx.setsockopt(zmq.LINGER, args.timeout)
			with ctx.socket(zmq.PUSH) as frontend:
				try:
					frontend.connect('tcp://%s:%d' % (args.ip, args.port))
					frontend.send_multipart([b'0', ServerCmd.terminate, b'0', b'0'])
					print('shutdown signal sent to %d' % args.port)
				except zmq.error.Again:
					raise TimeoutError(
						'no response from the server (with "timeout"=%d ms), please check the following:'
						'is the server still online? is the network broken? are "port" correct? ' % args.timeout)

	def run(self):
		self._run()

	@zmqd.context()
	@zmqd.socket(zmq.PULL)
	@zmqd.socket(zmq.PAIR)
	@multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
	def _run(self, _, frontend, sink, *backend_socks):

		def push_new_job(_job_id, _json_msg, _msg_len):
			# backend_socks[0] is always at the highest priority
			_sock = backend_socks[0] if _msg_len <= self.args.priority_batch_size else rand_backend_socket	# Smaller jobs are put on highest priority 
			_sock.send_multipart([_job_id, _json_msg])

		# bind all sockets
		self.logger.info('bind all sockets')
		frontend.bind('tcp://*:%d' % self.port)						# Ventilator socket, where users will send their requests
		addr_front2sink		= auto_bind(sink)					# It creates a socket file in some temporary folder which is used for Inter process commucation
		addr_backend_list	= [auto_bind(b) for b in backend_socks]			# Create self.num_concurrent_socket number of IPC files. Each entry is a string which looks like this: 'ipc://tmpm9Gxig/socket'
		self.logger.info('open %d ventilator-worker sockets' % len(addr_backend_list))

		# start the sink process | The function of sink is to PULL the output from workers and PUB it back to the users
		self.logger.info('start the sink')
		proc_sink = MedTypeSink(self.args, addr_front2sink)
		self.processes.append(proc_sink)
		proc_sink.start()
		addr_sink = sink.recv().decode('ascii')		# Sink will receive data from this address

		# start the backend processes
		device_map = self._get_device_map()
		for idx, device_id in enumerate(device_map):
			process = MedTypeWorkers(idx, self.args, addr_backend_list, addr_sink, device_id, self.model_params, self.umls2type, self.type2id)
			self.processes.append(process)
			process.start()

		# start the http-service process
		if self.args.http_port:
			self.logger.info('start http proxy')
			proc_proxy = MedTypeHTTPProxy(self.args)
			self.processes.append(proc_proxy)
			proc_proxy.start()

		rand_backend_socket = None
		server_status = ServerStatistic()

		for p in self.processes:
			p.is_ready.wait()

		self.is_ready.set()
		self.logger.info('all set, ready to serve request!')

		while True:
			try:
				request = frontend.recv_multipart()
				client, msg, req_id, msg_len = request
				assert req_id.isdigit()
				assert msg_len.isdigit()
			except (ValueError, AssertionError):
				self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
				self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
			else:
				server_status.update(request)
				if msg == ServerCmd.terminate:
					break
				elif msg == ServerCmd.show_config or msg == ServerCmd.show_status:
					self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client))
					status_runtime = {
							'client'		: client.decode('ascii'),
							'num_process'		: len(self.processes),
							'ventilator -> worker'	: addr_backend_list,
							'worker -> sink'	: addr_sink,
							'ventilator <-> sink'	: addr_front2sink,
							'server_current_time'	: str(datetime.now()),
							'device_map'		: device_map,
							'num_concurrent_socket'	: self.num_concurrent_socket
						}
					if msg == ServerCmd.show_status:
						status_runtime['statistic'] = server_status.value
					sink.send_multipart([client, msg, jsonapi.dumps({**status_runtime, **self.status_args, **self.status_static}), req_id])
				else:
					self.logger.info('new encode request\treq id: %d\tsize: %d\tclient: %s' % (int(req_id), int(msg_len), client))

					# register a new job at sink
					sink.send_multipart([client, ServerCmd.new_job, msg_len, req_id])

					# renew the backend socket to prevent large job queueing up
					# [0] is reserved for high priority job
					# last used backennd shouldn't be selected either as it may be queued up already
					rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])

					# push a new job, note super large job will be pushed to one socket only,
					# leaving other sockets free
					job_id = client + b'#' + req_id
					if int(msg_len) > self.max_batch_size:
						seqs	= jsonapi.loads(msg)
						job_gen	= ((job_id + b'@%d' % i, seqs[i:(i + self.max_batch_size)]) for i in range(0, int(msg_len), self.max_batch_size))  # Did batching if the size exceed maximum batch size
						for partial_job_id, job in job_gen:
							push_new_job(partial_job_id, jsonapi.dumps(job), len(job))
					else:
						push_new_job(job_id, msg, int(msg_len))

		for p in self.processes:
			p.close()
		self.logger.info('terminated!')

	def _get_device_map(self):
		self.logger.info('get devices')
		run_on_gpu = False
		device_map = [-1] * self.num_worker
		if not self.args.cpu:
			try:
				import GPUtil
				num_all_gpu	= len(GPUtil.getGPUs())
				avail_gpu	= GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker), maxMemory=0.9, maxLoad=0.9)
				num_avail_gpu	= len(avail_gpu)

				if num_avail_gpu >= self.num_worker:
					run_on_gpu = True
				elif 0 < num_avail_gpu < self.num_worker:
					self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' % (num_avail_gpu, num_all_gpu, self.num_worker))
					if not self.args.device_map:
						self.logger.warning('multiple workers will be allocated to one GPU, may not scale well and may raise out-of-memory')
					else:
						self.logger.warning('workers will be allocated based on "-device_map=%s", may not scale well and may raise out-of-memory' % self.args.device_map)
					run_on_gpu = True
				else:
					self.logger.warning('no GPU available, fall back to CPU')

				if run_on_gpu:
					device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
			except FileNotFoundError:
				self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. fall back to cpu!')
		self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
			'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
			enumerate(device_map)))
		return device_map


class MedTypeSink(Process):
	def __init__(self, args, front_sink_addr):
		super().__init__()
		self.port			= args.port_out						# Port at which results will be PUB for users to get response
		self.exit_flag			= multiprocessing.Event()				# An event for synchronizing threads
		self.logger			= set_logger(colored('SINK', 'green'), args.verbose)	# Logger
		self.verbose			= args.verbose    					# Set verbose level
		self.front_sink_addr		= front_sink_addr					# Ventilator <--> sink connection
		self.max_seq_len		= args.max_seq_len 					# Model related arguments
		self.is_ready			= multiprocessing.Event()				# Even for synchronizing threads
		self.max_position_embeddings	= 0

	def close(self):
		self.logger.info('shutting down...')
		self.is_ready.clear()
		self.exit_flag.set()
		self.terminate()
		self.join()
		self.logger.info('terminated!')

	def run(self):
		self._run()

	@zmqd.socket(zmq.PULL)
	@zmqd.socket(zmq.PAIR)
	@zmqd.socket(zmq.PUB)
	def _run(self, receiver, frontend, sender):
		receiver_addr = auto_bind(receiver) 		# Receiver will read from a created temporary file
		frontend.connect(self.front_sink_addr)		# Paired with Ventilator
		sender.bind('tcp://*:%d' % self.port)		# Results will be published on this port

		pending_jobs = ddict(lambda: SinkJob(self.max_seq_len))  # type: Dict[str, SinkJob]

		poller = zmq.Poller()
		poller.register(frontend, zmq.POLLIN)
		poller.register(receiver, zmq.POLLIN)

		# send worker receiver address back to frontend
		frontend.send(receiver_addr.encode('ascii'))

		# Windows does not support logger in MP environment, thus get a new logger
		# inside the process for better compability
		logger = set_logger(colored('SINK', 'green'), self.verbose)
		logger.info('ready')
		self.is_ready.set()

		while not self.exit_flag.is_set():
			socks = dict(poller.poll())

			# PULLING data from workers
			if socks.get(receiver) == zmq.POLLIN:
				msg		= receiver.recv_multipart()
				job_id		= msg[0]

				# parsing job_id and partial_id
				job_info	= job_id.split(b'@')
				job_id		= job_info[0]
				partial_id	= int(job_info[1]) if len(job_info) == 2 else 0

				if msg[2] == ServerCmd.elink_out:	# 'TOKENS'
					x = jsonapi.loads(msg[1])				# Get the tokens 
					pending_jobs[job_id].add_token(x, partial_id)		# Update the obtained value in pending_jobs
				else:
					logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(msg))
					logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(msg)), exc_info=True)

				logger.info('collect %s %s (E:%d/A:%d)' % (msg[2], job_id, pending_jobs[job_id].progress_tokens, pending_jobs[job_id].checksum))

			if socks.get(frontend) == zmq.POLLIN:
				client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()

				if msg_type == ServerCmd.new_job:
					job_info = client_addr + b'#' + req_id

					# register a new job
					pending_jobs[job_info].checksum = int(msg_info)
					logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
					if len(pending_jobs[job_info]._pending_embeds) > 0 and pending_jobs[job_info].final_ndarray is None:
						pending_jobs[job_info].add_embed(None, 0)

				elif msg_type == ServerCmd.show_config or msg_type == ServerCmd.show_status:
					time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
					logger.info('send config\tclient %s' % client_addr)
					sender.send_multipart([client_addr, msg_info, req_id])

			# check if there are finished jobs, then send it back to workers
			finished = [(k, v) for k, v in pending_jobs.items() if v.is_done]
			for job_info, tmp in finished:
				client_addr, req_id	= job_info.split(b'#')
				sender.send_multipart([client_addr, tmp.result, req_id])
				logger.info('send back\tsize: %d\tjob id: %s' % (tmp.checksum, job_info))
				# release the job
				tmp.clear()
				pending_jobs.pop(job_info)


class SinkJob:
	def __init__(self, max_seq_len):
		self._pending_embeds		= []
		self.elinks 			= []
		self.elinks_ids 		= []
		self.checksum			= 0
		self.progress_tokens		= 0
		self.max_effective_len		= 0

	def clear(self):
		self._pending_embeds.clear()
		self.elinks.clear()
		self.elinks_ids.clear()

	# Binary search for inserting tokens
	def _insert(self, data, pid, data_lst, idx_lst):
		lo = 0
		hi = len(idx_lst)
		while lo < hi:
			mid = (lo + hi) // 2
			if pid < idx_lst[mid]:
				hi = mid
			else:
				lo = mid + 1

		idx_lst.insert(lo, pid)
		data_lst.insert(lo, data)

	def add_token(self, data, pid):
		progress = len(data)
		self._insert(data, pid, self.elinks, self.elinks_ids)
		self.progress_tokens += progress

	@property
	def is_done(self):
		return self.checksum > 0 and self.checksum == self.progress_tokens

	@property
	def result(self):
		x = jsonapi.dumps(self.elinks)
		return x

class MedTypeWorkers(Process):

	def __init__(self, id, args, worker_address_list, sink_address, device_id, model_params, umls2type, type2id):
		super().__init__()
		self.args 			= args
		self.worker_id			= id
		self.device_id			= device_id
		self.logger			= set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), args.verbose)
		self.max_seq_len		= args.max_seq_len
		self.do_lower_case		= args.do_lower_case
		self.daemon			= True
		self.exit_flag			= multiprocessing.Event()
		self.worker_address		= worker_address_list
		self.num_concurrent_socket	= len(self.worker_address)
		self.sink_address		= sink_address
		self.prefetch_size		= args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
		self.gpu_memory_fraction	= args.gpu_memory_fraction
		self.model_path			= args.model_path
		self.model_type			= args.model_type
		self.model_params		= model_params
		self.entity_linker 		= args.entity_linker
		self.dropout 			= args.dropout
		self.verbose			= args.verbose
		self.tokenizer_model 		= args.tokenizer_model
		self.context_len 		= args.context_len
		self.batch_size 		= args.model_batch_size
		self.threshold 			= args.threshold
		self.is_ready			= multiprocessing.Event()
		self.ent_linker 		= self.get_linkers(args.entity_linker)
		self.umls2type 			= umls2type
		self.type2id 			= type2id
		self.id2type			= {v: k for k, v in self.type2id.items()}

		self.tokenizer	= BertTokenizer.from_pretrained(self.tokenizer_model)
		self.tokenizer.add_tokens(['[MENTION]', '[/MENTION]'])
		self.cls_tok	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS]'))
		self.sep_tok	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'))
		self.men_start  = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[MENTION]'))
		self.men_end 	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[/MENTION]'))

	
	def get_ent_linker(self, linker):
		if   linker.lower() == 'scispacy': 	return ScispaCy(self.args)
		elif linker.lower() == 'quickumls': 	return QUMLS(self.args)
		elif linker.lower() == 'ctakes': 	return CTakes(self.args)
		elif linker.lower() == 'metamap': 	return Metamap(self.args)
		elif linker.lower() == 'metamaplite': 	return MetamapLite(self.args)
		else: raise NotImplementedError

	def get_linkers(self, linkers_list):
		ent_linkers = {}
		for linker in linkers_list.split(','):
			ent_linkers[linker] = self.get_ent_linker(linker)
		return ent_linkers


	def close(self):
		self.logger.info('shutting down...')
		self.exit_flag.set()
		self.is_ready.clear()
		self.terminate()
		self.join()
		self.logger.info('terminated!')

	def run(self):
		self._run()

	def pad_data(self, data):
		max_len  = np.max([len(x['toks']) for x in data])
		tok_pad	 = np.zeros((len(data), max_len), np.int32)
		tok_mask = np.zeros((len(data), max_len), np.float32)
		men_pos  = np.zeros((len(data)), np.int32)
		meta 	 = []

		for i, ele in enumerate(data):
			tok_pad[i, :len(ele['toks'])]   = ele['toks']
			tok_mask[i, :len(ele['toks'])]  = 1.0
			men_pos[i] 			= ele['men_pos'] 
			meta.append({
				'text_id': ele['text_id'],
				'men_id' : ele['men_id']
			})

		return torch.LongTensor(tok_pad).to(self.device), torch.FloatTensor(tok_mask).to(self.device), torch.LongTensor(men_pos).to(self.device), meta

	def get_batches(self, elinks):
		data_list = []

		for t_id, ele in enumerate(elinks):
			text = ele['text']

			for m_id, men in enumerate(ele['mentions']):
				start, end = men['start_offset'], men['end_offset']

				mention   = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[start:end]))
				prev_toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[:start]))[-self.context_len//2:]
				next_toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[end:]))[:self.context_len//2]
				toks	  = self.cls_tok + prev_toks + self.men_start + mention + self.men_end + next_toks + self.sep_tok

				data_list.append({
					'text_id'	: t_id,
					'men_id'	: m_id,
					'toks'		: toks,
					'men_pos'	: len(prev_toks) + 1
				})

		num_batches = int(np.ceil(len(data_list) / self.batch_size))
		for i in range(num_batches):
			start_idx = i * self.batch_size
			yield self.pad_data(data_list[start_idx : start_idx + self.batch_size])

	def filter_candidates(self, elinks):
		out = ddict(lambda: ddict(dict))

		with torch.no_grad():

			for batch in self.get_batches(elinks):

				logits = self.model(
						input_ids	= batch[0], 
						attention_mask 	= batch[1],
						mention_pos_idx = batch[2]
					)

				preds = (torch.sigmoid(logits) > self.threshold).cpu().numpy()

				for i, ele in enumerate(batch[3]):
					out[ele['text_id']][ele['men_id']] = set([self.id2type[x] for x in np.where(preds[i])[0]])

		filt_elinks = []
		for t_id, ele in enumerate(elinks):
			mentions = []
			for m_id, men in enumerate(ele['mentions']):
				men['pred_type'] = out[t_id][m_id]

				# No filtering when predicted type is NA
				if len(men['pred_type']) == 0:  
					men['filtered_candidates'] = men['candidates']
				else: 				
					men['filtered_candidates'] = [[cui, scr] for cui, scr in men['candidates'] if len(self.umls2type.get(cui, set()) & men['pred_type']) != 0]
					if len(men['filtered_candidates']) == 0:
						men['filtered_candidates'] = men['candidates']

				men['pred_type'] = list(men['pred_type'])	# set is not JSON serializable
				mentions.append(men)

			ele['mentions'] = mentions
			filt_elinks.append(ele)

		return filt_elinks

	@zmqd.socket(zmq.PUSH)
	@zmqd.socket(zmq.PUSH)
	@multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
	def _run(self, sink_embed, sink_token, *receivers):
		# Windows does not support logger in MP environment, thus get a new logger
		# inside the process for better compatibility
		logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)
		logger.info('use device %s, load graph from %s' % ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.model_path))

		for sock, addr in zip(receivers, self.worker_address):
			sock.connect(addr)

		sink_embed.connect(self.sink_address)
		sink_token.connect(self.sink_address)

		poller = zmq.Poller()
		for sock in receivers:
			poller.register(sock, zmq.POLLIN)

		logger.info('ready and listening!')
		self.is_ready.set()
		
		if str(self.device_id) != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda:{}'.format(self.device_id))
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		if   self.model_type.lower() == 'bert_combined': self.model = BertCombined(len(self.tokenizer), len(self.type2id), self.dropout)
		elif self.model_type.lower() == 'bert_plain':    self.model = BertPlain(len(self.tokenizer), len(self.type2id), self.dropout)
		else: raise NotImplementedError

		self.model.to(self.device)
		self.model.load_state_dict(self.model_params, strict=False)

		logger	= set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)
		while not self.exit_flag.is_set():
			events = dict(poller.poll())
			for sock_idx, sock in enumerate(receivers):
				if sock in events:
					client_id, raw_msg	= sock.recv_multipart()
					message			= jsonapi.loads(raw_msg)
					logger.info('new job\tsocket: %d\tsize: %d\tlinker: %s\tclient: %s' % (sock_idx, len(message['text']), message['entity_linker'], client_id))

					if message['entity_linker'] in self.ent_linker:
						elinks = []
						for text in message['text']:
							elinks.append(self.ent_linker[message['entity_linker']](text))
						filt_elinks = self.filter_candidates(elinks)
					else:
						logger.info('Requested linker %s from \tsocket: %d\tclient: %s not loaded on server' % (message['entity_linker'], sock_idx, client_id))
						elinks = []
						filt_elinks = []

					sink_embed.send_multipart([client_id, jsonapi.dumps(filt_elinks), ServerCmd.elink_out], copy=True, track=False)
					logger.info('job done\tsize: %s\tclient: %s' % (len(filt_elinks), client_id))


class ServerStatistic:
	def __init__(self):
		self._hist_client		= CappedHistogram(500)
		self._hist_msg_len		= ddict(int)
		self._client_last_active_time	= CappedHistogram(500)
		self._num_data_req		= 0
		self._num_sys_req		= 0
		self._num_total_seq		= 0
		self._last_req_time		= time.perf_counter()
		self._last_two_req_interval	= []
		self._num_last_two_req		= 200

	def update(self, request):
		client, msg, req_id, msg_len = request
		self._hist_client[client] += 1
		if ServerCmd.is_valid(msg):
			self._num_sys_req += 1
			# do not count for system request, as they are mainly for heartbeats
		else:
			self._hist_msg_len[int(msg_len)]	+= 1
			self._num_total_seq			+= int(msg_len)
			self._num_data_req			+= 1
			tmp = time.perf_counter()
			self._client_last_active_time[client] = tmp
			if len(self._last_two_req_interval) < self._num_last_two_req:
				self._last_two_req_interval.append(tmp - self._last_req_time)
			else:
				self._last_two_req_interval.pop(0)
			self._last_req_time = tmp

	@property
	def value(self):
		def get_min_max_avg(name, stat, avg=None):
			if len(stat) > 0:
				avg = sum(stat) / len(stat) if avg is None else avg
				min_, max_ = min(stat), max(stat)
				return {
					'avg_%s' % name: avg,
					'min_%s' % name: min_,
					'max_%s' % name: max_,
					'num_min_%s' % name: sum(v == min_ for v in stat),
					'num_max_%s' % name: sum(v == max_ for v in stat),
				}
			else:
				return {}

		def get_num_active_client(interval=180):
			# we count a client active when its last request is within 3 min.
			now = time.perf_counter()
			return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)

		avg_msg_len = None
		if len(self._hist_msg_len) > 0:
			avg_msg_len = sum(k*v for k,v in self._hist_msg_len.items()) / sum(self._hist_msg_len.values())

		parts = [{
			'num_data_request'	: self._num_data_req,
			'num_total_seq'		: self._num_total_seq,
			'num_sys_request'	: self._num_sys_req,
			'num_total_request'	: self._num_data_req + self._num_sys_req,
			'num_total_client'	: self._hist_client.total_size(),
			'num_active_client'	: get_num_active_client()},
			self._hist_client.get_stat_map('request_per_client'),
			get_min_max_avg('size_per_request', self._hist_msg_len.keys(), avg=avg_msg_len),
			get_min_max_avg('last_two_interval', self._last_two_req_interval),
			get_min_max_avg('request_per_second', [1. / v for v in self._last_two_req_interval]),
		]

		return {k: v for d in parts for k, v in d.items()}
