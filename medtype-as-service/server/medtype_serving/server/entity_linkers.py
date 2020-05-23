from collections import defaultdict as ddict
import re, requests
from .helper import *

class BaseLinker:

	def reformat(self, men_list, text):
		out = {
			'text'	   : text,
			'mentions' : []
		}

		for ((start, end), cands) in men_list.items():
			out['mentions'].append({
				'mention'	: text[start: end],
				'start_offset'	: start,
				'end_offset'	: end,
				'candidates'	: cands
			})

		return out

######################### SCISPACY

class ScispaCy(BaseLinker):

	def __init__(self, args):
		import scispacy, spacy
		from scispacy.abbreviation import AbbreviationDetector
		from scispacy.umls_linking import UmlsEntityLinker

		self.nlp = spacy.load("en_core_sci_sm")
		self.nlp.add_pipe(AbbreviationDetector(self.nlp))	# Add abbreviation deteciton module
		linker 	= UmlsEntityLinker(resolve_abbreviations=True) 	# Add Entity linking module
		self.nlp.add_pipe(linker)

	def __call__(self, text):
		sci_res	 = self.nlp(text)
		men_list = ddict(list)

		for ent in sci_res.ents:
			start, end = ent.start_char, ent.end_char
			for cand, score in ent._.umls_ents:
				men_list[(start, end)].append([cand, round(score, 3)])

		return self.reformat(men_list, text)

######################### QUICK-UMLS

class QUMLS(BaseLinker):

	def __init__(self, args):
		from quickumls import QuickUMLS

		assert args.quickumls_path is not None, "Please provide path where QuickUMLS is installed"
		assert args.num_worker == 1, "QuickUMLS doesn't support num_workers > 1"

		self.matcher = QuickUMLS(args.quickumls_path, 'score', threshold=0.6)

	def __call__(self, text):
		qumls_res = self.matcher.match(text)
		men_list  = ddict(list)
		for men in qumls_res:
			for cand in men:
				start, end = cand['start'], cand['end']
				umls_cui   = cand['cui']
				score 	   = cand['similarity']
				men_list[(start, end)].append([umls_cui, round(score, 3)])

		return self.reformat(men_list, text)

######################### cTAKES

class CTakes(BaseLinker):

	def __init__(self, args):
		# The ctakes server is running with the following command in $CTAKES_HOME/ctakes_server
		# java -Dctakes.umlsuser=XXXX -Dctakes.umlspw=XXXX -Xmx5g -cp -target/ctakes-server-0.1.jar:resources/ de.dfki.lt.ctakes.Server localhost 9898 desc/ctakes-clinical-pipeline/desc/analysis_engine/AggregatePlaintextUMLSProcessor.xml
		self.ctakes_url  = 'http://{}:{}/ctakes'.format(args.ctakes_host, args.ctakes_port)
		assert args.num_worker == 1, "cTakes doesn't support num_workers > 1"

	def __call__(self, text):
		import requests
		try:
			text = clean_text(text)
			res  = requests.get(self.ctakes_url, params={"text": text});

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
				return self.reformat(res_list, text)

		except Exception as e:
			print('\nException in cTakes: {}'.format(e.args[0]))


		return self.reformat({}, text)


######################### MetaMap

class Metamap(BaseLinker):

	def __init__(self, args):
		from pymetamap import MetaMap
		self.model  = MetaMap.get_instance(args.metamap_path)
	
	def __call__(self, text):
		text = clean_text(text)
		concepts, error  = self.model.extract_concepts([text], [1])

		res_list = ddict(list)
		for k, concept in enumerate(concepts):
			if concept[1] !='MMI': continue

			pos_info = [list(map(int, x.split('/'))) for x in concept.pos_info.replace(',', ';').replace('[', '').replace(']', '').split(';')]
			men_cnt  = [len(x.split(',')) for x in concept.pos_info.split(';')]
			men_sing = replace(concept.trigger, '"').split('"')[1::2][1::2]
			mentions = mergeList([[men]*men_cnt[j] for j, men in enumerate(men_sing)])

			for i, (start, offset) in enumerate(pos_info):
				start = start - 1
				end = start + offset
				res_list[(start, end)].append((concept.cui, concept.score))

		return self.reformat(res_list, text)

######################### MetaMapLite

class MetamapLite(BaseLinker):

	def __init__(self, args):
		from pymetamap import MetaMapLite
		self.model  = MetaMapLite.get_instance(args.metamaplite_path)
	
	def __call__(self, text):
		text = clean_text(text)
		concepts, error  = self.model.extract_concepts([text], [1])

		res_list = ddict(list)
		for k, concept in enumerate(concepts):
			if concept.mm !='MMI': continue

			pos_info = [list(map(int, x.split('/'))) for x in concept.pos_info.split(';')]
			mentions = replace(concept.trigger, '"').split('"')[0::2][1::2]

			for i, (start, offset) in enumerate(pos_info):
				start = start - 1
				end = start + offset
				res_list[(start, end)].append((concept.cui, concept.score))

		return self.reformat(res_list, text)