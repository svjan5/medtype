"""
Dumps the datasets to be evaluated in the format required for MedType to predict the type. 
"""

from helper import *
from nltk.tokenize import word_tokenize

######################### Reads MongoDB and generates the dataset for training pred_type.py

def get_type_data():

	umls2type = load_pickle('./data/umls2type.pkl')

	def process_data(pid, doc_list):

		data = []
		for k, (doc, dataset) in enumerate(doc_list):

			tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
			text 	  = doc['text']

			for men in doc['mentions']:
				start, end = men['start_offset'], men['end_offset']

				mention    = text[start: end]
				prev_toks  = word_tokenize(text[:start])[-args.con_len:]
				after_toks = word_tokenize(text[end:])[:args.con_len]
				type_label = list(set.union(*[umls2type.get(x, set()) for x in men['link_id'].split('|')]))

				ele = {
					'mention'		: mention,
					'mention_toks_bert'	: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(mention))),
					'prev_toks'		: prev_toks,
					'prev_toks_bert'	: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(prev_toks))),
					'after_toks'		: after_toks,
					'after_toks_bert'	: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(after_toks))),
					'label'			: type_label,
					'start'			: start,
					'end'			: end,
					'_id'			: dataset + '_' + doc['_id'],
					'split'			: doc['split'],
				}

				data.append(ele)

			if k % 10 == 0: print('Completed [{}] {}, {}'.format(pid, k, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
			
		return data

	doc_list  = []
	for data in args.data.split(','):
		for line in open('../datasets/{}.json'.format(data)):
			doc = json.loads(line.strip())
			doc_list.append((doc, data))

	num_procs = args.workers
	chunks    = partition(doc_list, num_procs)
	data_list = mergeList(Parallel(n_jobs = num_procs)(delayed(process_data)(i, chunk) for i, chunk in enumerate(chunks)))

	dump_pickle(data_list, './data/{}.pkl'.format(args.data))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data', 	 	default='medmentions')
	parser.add_argument('--con_len', 	default=100, 	type=int)
	parser.add_argument('--workers', 	default=4, 	type=int)
	args = parser.parse_args()

	get_type_data()