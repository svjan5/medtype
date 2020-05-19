from helper import *
from joblib import Parallel, delayed
import requests, re


######################### Dump Ground Truth in required format for evaluation

def groundtruth_dump(doc_list):
	base_dir = './results/{}'.format(args.data); make_dir(base_dir)
	fname	 = './results/{}/ground_{}.txt'.format(args.data, args.split)

	if not checkFile(fname): 
		writer = csv.writer(open(fname, 'w'), delimiter='\t')
		for doc in doc_list:
			for men in doc['mentions']:
				writer.writerow([doc['_id'], men['start_offset'], men['end_offset'], men['link_id'], 1.0, 'O'])

######################### SCISPACY

def scispacy(doc_list):
	import scispacy, spacy
	from scispacy.abbreviation import AbbreviationDetector
	from scispacy.umls_linking import UmlsEntityLinker

	def process_data(pid, doc_list):
		nlp 	= spacy.load("en_core_sci_sm")
		nlp.add_pipe(AbbreviationDetector(nlp))						# Add abbreviation deteciton module
		linker 	= UmlsEntityLinker(resolve_abbreviations=True); nlp.add_pipe(linker)	# Add Entity linking module

		data = []
		for i, doc in enumerate(doc_list):
			sci_res	 = nlp(doc['text'])
			res_list = {}

			for ent in sci_res.ents:
				start, end = ent.start_char, ent.end_char
				res_list[(start, end)] = ent._.umls_ents

			doc['result'] = res_list
			data.append(doc)

			if i % 10 == 0: 
				print('Completed [{}] {}, {}'.format(pid, i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
		
		return data
	
	num_procs = args.workers
	chunks    = partition(doc_list, num_procs)
	data_list = mergeList(Parallel(n_jobs = num_procs)(delayed(process_data)(i, chunk) for i, chunk in enumerate(chunks)))

	base_dir  = './results/{}'.format(args.data); make_dir(base_dir)
	dump_pickle(data_list, '{}/{}_{}.pkl'.format(base_dir, args.model, args.split))

######################### QUICK-UMLS

def quickumls(doc_list):
	from quickumls import QuickUMLS

	assert not args.quickumls_path is None, "Provide path where QuickUMLS is installed"

	def process_data(pid, doc_list):
		data = []
		matcher	= QuickUMLS(args.quickumls_path, 'score', threshold=0.6)
		for i, doc in enumerate(doc_list):
			qumls_res = matcher.match(doc['text'])

			res_list  = ddict(list)
			for men in qumls_res:
				for cand in men:
					start, end = cand['start'], cand['end']
					umls_cui   = cand['cui']
					score 	   = cand['similarity']
					res_list[(start, end)].append((umls_cui, score))

			doc['result'] = dict(res_list)
			data.append(doc)

			if i % 10 == 0: 
				print('Completed [{}] {}, {}'.format(pid, i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
		
		return data
	
	num_procs = 1
	chunks    = partition(doc_list, num_procs)
	data_list = mergeList(Parallel(n_jobs = num_procs)(delayed(process_data)(i, chunk) for i, chunk in enumerate(chunks)))

	base_dir  = './results/{}'.format(args.data); make_dir(base_dir)
	dump_pickle(data_list, '{}/{}_{}.pkl'.format(base_dir, args.model, args.split))

######################### CTAKES

def ctakes(doc_list):
	import ctakes_parser as cparser

	ctakes_url  = 'http://{}:{}/ctakes'.format(args.ctakes_host, args.ctakes_port)

	def process_data(pid, doc_list):
		data = []
		for i, doc in enumerate(doc_list):
			text = clean_text(doc['text'])
			res  = requests.get(ctakes_url, params={"text": text});

			res_list = ddict(set)
			if res.status_code == 200:
				data = res.json()
				for dat in data:
					if 'ontologyConceptArr' in dat['annotation'] and not dat['annotation']['ontologyConceptArr'] is None:
						start = dat['annotation']['begin']
						end   = dat['annotation']['end']
						for cands in dat['annotation']['ontologyConceptArr']:
							umls_cui = cands['annotation']['cui']
							score    = cands['annotation']['score']			# 0.0 as lookup
							res_list[(start, end)].add((umls_cui, round(score, 3)))
				
				res_list = {k: list(v) for k, v in res_list.items()}
				doc['result'] = dict(res_list)
				data.append(doc)
			
			if i % 10 == 0: 
				print('Completed [{}] {}, {}'.format(pid, i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))


		return data

	num_procs = args.workers
	chunks    = partition(doc_list, num_procs)
	data_list = mergeList(Parallel(n_jobs = num_procs)(delayed(process_data)(i, chunk) for i, chunk in enumerate(chunks)))

	base_dir  = './results/{}'.format(args.data); make_dir(base_dir)
	dump_pickle(data_list, '{}/{}_{}.pkl'.format(base_dir, args.model, args.split))

######################### METAMAP

def metamap(doc_list):
	from pymetamap import MetaMap
	mm = MetaMap.get_instance(args.metamap_path)

	def process_data(pid, doc_list):

		data = []
		for i, doc in enumerate(doc_list):

			try:
				text = clean_text(doc['text'])
				concepts, error  = mm.extract_concepts([text],[doc['_id']])
				
				assert len(text) == len(doc['text']), 'Text length does not match after pre-processing'

				res_list = ddict(list)
				for k, concept in enumerate(concepts):
					if concept[1] !='MMI': continue

					pos_info = [list(map(int, x.split('/'))) for x in concept.pos_info.replace(',', ';').replace('[', '').replace(']', '').split(';')]
					men_cnt  = [len(x.split(',')) for x in concept.pos_info.split(';')]
					men_sing = replace(concept.trigger, '"').split('"')[1::2][1::2]
					mentions = mergeList([[men]*men_cnt[j] for j, men in enumerate(men_sing)])

					for j, (start, offset) in enumerate(pos_info):						
						end = start + offset
						res_list[(start, end)].append((concept.cui, concept.score))

				doc['result'] = dict(res_list)
				data.append(doc)
				if i % 10 == 0: 
					print('Completed [{}] {}, {}'.format(pid, i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
			except Exception as e:
				print('\nException Cause: {}'.format(e.args[0]))
				continue
			

		print('All work done {}!!'.format(pid))
		return data

	num_procs = args.workers
	chunks    = partition(doc_list, num_procs)
	data_list = mergeList(Parallel(n_jobs= num_procs)(delayed(process_data)(i,chunk) for i, chunk in enumerate(chunks)))

	base_dir  = './results/{}'.format(args.data); make_dir(base_dir)
	dump_pickle(data_list, '{}/{}_{}.pkl'.format(base_dir, args.model, args.split))


######################### METAMAP LITE

def metamaplite(doc_list):
	from pymetamap import MetaMapLite
	mm = MetaMapLite.get_instance(args.metamaplite_path)

	def process_data(pid, doc_list):

		data, miss = [], 0
		for i, doc in enumerate(doc_list):
			try:
				text = clean_text(doc['text'])
				concepts, error  = mm.extract_concepts([text],[doc['_id']])
				assert len(text) == len(doc['text']), 'Text length does not match after pre-processing'

				res_list = ddict(list)
				for k, concept in enumerate(concepts):
					if concept.mm !='MMI': continue

					pos_info = [list(map(int, x.split('/'))) for x in concept.pos_info.split(';')]
					mentions = replace(concept.trigger, '"').split('"')[0::2][1::2]

					for j, (start, offset) in enumerate(pos_info):						
						end = start + offset
						res_list[(start, end)].append((concept.cui, concept.score))

				doc['result'] = dict(res_list)
				data.append(doc)
				if i % 10 == 0: 
					print('Completed [{}] {}, {}'.format(pid, i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))

			except Exception as e:
				print('\nException Cause: {}'.format(e.args[0]))
				miss += 1
				continue

		print('All work done {} | Miss: {}!!'.format(pid, miss))
		return data

	num_procs = args.workers
	chunks    = partition(doc_list, num_procs)
	data_list = mergeList(Parallel(n_jobs= num_procs)(delayed(process_data)(i,chunk) for i, chunk in enumerate(chunks)))

	base_dir  = './results/{}'.format(args.data); make_dir(base_dir)
	dump_pickle(data_list, '{}/{}_{}.pkl'.format(base_dir, args.model, args.split))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data', 		 default='ncbi', 	help='Dataset on which evaluation has to be performed.')
	parser.add_argument('--model', 		 default='metamap', 	help='Entity linking system to use. Options: [scispacy, quickumls, ctakes, metamamp, metamaplite]')
	parser.add_argument('--split', 		 default='test', 	help='Dataset split to evaluate on')
	parser.add_argument('--quickumls_path',  default=None, 		help='QuickUMLS installation directory')
	parser.add_argument('--metamap_path', 	 default=None, 		help='Location where MetaMap executable is installed, e.g .../public_mm/bin/metamap18')
	parser.add_argument('--metamaplite_path',default=None, 		help='Location where MetaMapLite is installed, e.g .../public_mm_lite')
	parser.add_argument('--ctakes_host', 	 default='localhost', 	help='IP at which cTakes server is running')
	parser.add_argument('--ctakes_port', 	 default=9999,type=int, help='Port at which cTakes server is running')
	parser.add_argument('--workers', 	 default=1,   type=int, help='Number of processes to use for parallelization')
	args = parser.parse_args()

	# Reading dataset
	doc_list = []
	for line in open('../datasets/{}.json'.format(args.data)):
		doc = json.loads(line.strip())		
		if doc['split'] != args.split: continue
		doc_list.append(doc)

	# Dump Ground truth
	groundtruth_dump(doc_list)

	# Dump Model's output
	if 	args.model == 'quickumls':	quickumls(doc_list)
	elif 	args.model == 'scispacy':	scispacy(doc_list)
	elif    args.model == 'ctakes':		ctakes(doc_list)
	elif 	args.model == 'metamap':	metamap(doc_list)
	elif 	args.model == 'metamaplite':	metamaplite(doc_list)
	else:	raise NotImplementedError
