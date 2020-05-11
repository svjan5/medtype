from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction, BertForSequenceClassification

class BertDataset(Dataset):
	def __init__(self, dataset, num_class, params):
		self.dataset	= dataset
		self.num_class 	= num_class
		self.p 		= params

		self.tokenizer 	= BertTokenizer.from_pretrained(params.bert_model)
		self.cls_tok 	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS]'))
		self.sep_tok 	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'))

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		return ele['prev_toks_bert'], ele['after_toks_bert'], ele['mention_toks_bert'], ele['label'], ele

	def pad_data(self, sents):
		max_len  = min(np.max([len(x[0])+len(x[1])+len(x[2]) for x in sents]), self.p.max_seq_len) 		# +2 for ['CLS'] and ['SEP'] 

		tok_pad	 = np.zeros((len(sents), max_len), np.int32)
		tok_pos  = np.zeros((len(sents), max_len), np.int32)
		tok_mask = np.zeros((len(sents), max_len), np.float32)
		labels	 = np.zeros((len(sents), self.num_class), np.int32)
		tok_len  = np.zeros((len(sents)), np.int32)

		for i, (prev, after, mention, label, _) in enumerate(sents):
			tot_len = len(prev) + len(after) + len(mention)

			if tot_len > max_len-2:
				left_len	= (max_len - len(mention) - 2) // 2
				prev		= prev[-left_len:]
				after		= after[:left_len]

			prev 	= self.cls_tok + prev
			after  	= after + self.sep_tok
			toks	= prev + mention + after

			if len(toks) > max_len:
				return None, None, None, None, None

			tok_pad [i, :len(toks)]		 = toks
			tok_mask[i, :len(toks)]		 = 1.0
			tok_pos [i, :len(toks)]		 = np.concatenate([np.arange(len(prev)-1, -1, -1), [0]*len(mention), np.arange(len(after))])
			tok_len [i]			 = len(toks)
			labels  [i, np.int32(label)]	 = 1
			

		return torch.LongTensor(tok_pad), torch.FloatTensor(tok_mask), torch.LongTensor(tok_pos), torch.LongTensor(tok_len), torch.LongTensor(labels)

	def collate_fn(self, all_data):
		all_data.sort(key = lambda x: -(len(x[0])+len(x[1])+len(x[2])) )

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]

			tok_pad, tok_mask, tok_pos, tok_len, labels = self.pad_data(data)
			if tok_pad is None: continue

			batches.append ({
				'tok_pad'	: tok_pad,
				'tok_mask'	: tok_mask,
				'tok_pos'	: tok_pos,
				'tok_len'	: tok_len,
				'labels'	: labels,
				'_rest'		: [x[-1] for x in data],
			})

		return batches