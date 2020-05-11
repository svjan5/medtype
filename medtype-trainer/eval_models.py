from helper import *

def filter_on_cat(cands, act_cat):
	idx = -1

	for k, (umls, scr) in enumerate(cands):
		pred_cat = umls2type.get(umls, set())
		if len(pred_cat & act_cat) != 0:
			idx = k
			sel_cand = cands[idx][0]
			return sel_cand, idx, pred_cat

	return None, idx, None

def get_pred(model_res, _id):
	if    _id in model_res: 		return model_res[_id]
	elif (args.data+'_'+_id) in model_res: 	return model_res[args.data+'_'+_id]
	elif _id.split('_')[1] in model_res: 	return model_res[_id.split('_')[1]]
	else: raise NotImplementedError
	
def dump_results():
	pred_file	= load_pickle('./results/{}/{}_{}.pkl'.format(args.data, args.model, args.split))
	fname		= './results/{}/{}_{}_{}_{}.txt'.format(args.data, args.model, args.split, args.ent_disamb, args.pred_model)
	f		= open(fname, 'w')
	writer		= csv.writer(f, delimiter='\t')
	benefit, loss   = 0, 0

	uni_freq  = ddict(int)
	pair_freq = ddict(int)

	for doc in pred_file:
		if doc['split'] != args.split: continue

		act_men_idx  = [(x['start_offset'], x['end_offset']) for x in doc['mentions']]
		act_men_tag  = [x['link_id'] for x in doc['mentions']]
		act_char_lvl = ['O' for _ in range(len(doc['text']))]
		for j, tag in enumerate(act_men_tag):
			start, end = act_men_idx[j]
			for k in range(start, end):
				if k < len(doc['text']):
					act_char_lvl[k] = tag

		act_char_hot = np.int32([1 if x != 'O'  else 0 for x in act_char_lvl])

		if args.pred_model is not None:
			pred_type_char = [{'O'} for _ in range(len(doc['text']))]
			for (start, end), type_ids in get_pred(model_res, doc['_id']):
				for k in range(start, end):
					if k < len(doc['text']):
						pred_type_char[k] = {id2type[x] for x in type_ids}

		for (start, end), cands in doc['result'].items():
			if np.sum(act_char_hot[start: end]) == 0 and 'wiki' in args.data: continue

			act_cui	= mergeList([x.split('|') for x in set(act_char_lvl[start: end]) if x != 'O'])
			if len(act_cui) == 0: act_cui	= ['O']

			if   len(cands) == 0: 		sel_cand = 'O'			# No candidates to choose from
			elif not args.ent_disamb: 	sel_cand = cands[0][0]		# Default setting: Selected the first candidate
				
			elif args.pred_model is not None:				# Using Prediction model
				act_cat = set.union(*pred_type_char[start: end])

				if act_cat == {'O'}:
					sel_cand = cands[0][0]
				else:
					sel_cand, idx, _ = filter_on_cat(cands, act_cat)
					if idx == -1: sel_cand = cands[0][0]

			else:								# Oracle case
				if act_cui == ['O']: sel_cand = cands[0][0]

				else:	
					act_cat			= set.union(*[umls2type.get(cui, set()) for cui in act_cui])
					sel_cand, idx, pred_cat	= filter_on_cat(cands, act_cat)

					if idx == -1: sel_cand = cands[0][0]

			if sel_cand is not None:
				writer.writerow([doc['_id'], start, end, sel_cand, 1.0, 'O'])

	f.close()
	cmd    = 'cd neleval; ./nel evaluate -g ../results/{}/ground_{}.txt ../{} -m overlap-maxmax::span+kbid -m strong_all_match  -m sets::kbid'.format(args.data, args.split, fname)
	result = os.popen(cmd).read()
	return result

if __name__ == '__main__':

	parser 	= argparse.ArgumentParser(description='')
	parser.add_argument('--data', 		default='medmentions', 		help='Dataset on which evaluation has to be performed.')
	parser.add_argument('--model', 		default='scispacy', 		help='Entity linking system to use. Options: [scispacy, quickumls, ctakes, metamamp, metamaplite]')
	parser.add_argument('--split', 		default='test', 		help='Dataset split to evaluate on')
	parser.add_argument('--ent_disamb', 	action='store_true', 		help='Incorporate Entity Disambiguation step')
	parser.add_argument('--pred_model', 	default=None, 	type=str, 	help='Models prediction to use')
	parser.add_argument('--thresh', 	default=0.5, 	type=float, 	help='Threshold to use on logits')
	args		= parser.parse_args()

	if args.ent_disamb:
		type_remap	= json.load(open('../config/type_remap.json'.format(args.ent_disamb)))
		type2id		= json.load(open('../config/type2id.json'.format(args.ent_disamb)))
		id2type		= {v: k for k, v in type2id.items()}

		umls2type_fine	= load_pickle('./data/umls2type.pkl')
		umls2type	= {umls_id: set([type_remap[x] for x in types]) for umls_id, types in umls2type_fine.items()}

		if args.pred_model is not None:

			all_pred  = ddict(dict)
			mdl_preds = load_pickle('./predictions/{}/{}'.format(args.data, args.pred_model))

			logits	  = np.concatenate(mdl_preds['logits'], axis=0)
			labels	  = np.concatenate(mdl_preds['labels'], axis=0)
			others    = mergeList(mdl_preds['others'])
			pred_prb  = sigmoid(logits)

			for i, ele in enumerate(others):
				all_pred[ele['_id']][(ele['start'], ele['end'])] = pred_prb[i]
			
			model_res = ddict(list)
			for _id, vals in all_pred.items():
				for (start, end), prob in vals.items():
					pred = [x for x in np.where(pred)[0]]
					model_res[_id].append(((start, end), pred))

			model_res = dict(model_res)

	result = dump_results()
	print(result)