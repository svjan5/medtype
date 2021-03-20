from helper import *
from models import BertPlain, BertCombined
from dataloader import MedTypeDataset

from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from sklearn.metrics import average_precision_score

class MedType(object):

	def load_data(self):
		"""
		Reads in the data for training MedType.

		Parameters
		----------
		self.p.data:         	Takes in the name of the dataset  (FB15k-237, WN18RR, YAGO3-10)
		
		Returns
		-------
		self.type2id:           Semantic Type to unique identifier mapping
		self.data['train']:     Stores the training split of the dataset
		self.data['valid']:     Stores the validation split of the dataset
		self.data['test']:      Stores the test split of the dataset
		self.data_iter:		The dataloader for different data splits
		"""

		self.type2id	= json.load(open('{}/type2id.json'.format(self.p.config_dir)))
		self.num_class	= len(self.type2id)
		type_remap 	= json.load(open('{}/type_remap.json'.format(self.p.config_dir)))
		self.data 	= {'train': [], 'test': [], 'valid': []}

		if self.p.data == 'pubmed':
			for root, dirs, files in os.walk('./{}/pubmed_processed'.format(self.p.data_dir)):

				for file in tqdm(files):
					fname = os.path.join(root, file)
					for line in open(fname):
						doc 		= json.loads(line.strip())
						doc['label'] 	= list(set([self.type2id[type_remap[x]] for x in doc['label']]))
						del doc['prev_toks'], doc['after_toks']
						self.data['train'].append(doc)
			
			# In case of PubMedDS, test and valid split of Medmentions datasets is used.
			data = load_pickle('{}/medmentions.pkl'.format(self.p.data_dir))

			for doc in data:
				if doc['split'] in ['valid', 'test']:
					doc['label'] = list(set([self.type2id[type_remap[x]] for x in doc['label']]))
					del doc['prev_toks'], doc['after_toks']
					self.data[doc['split']].append(doc)

		else:
			data = load_pickle('{}/{}.pkl'.format(self.p.data_dir, self.p.data))

			for doc in data:
				doc['label'] = list(set([self.type2id[type_remap[x]] for x in doc['label']]))
				self.data[doc.get('split', 'train')].append(doc)

		self.logger.info('\nDataset size -- Train: {}, Valid: {}, Test:{}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])))

		self.tokenizer 	= BertTokenizer.from_pretrained(self.p.bert_model)
		self.tokenizer.add_tokens(['[MENTION]', '[/MENTION]'])

		def get_data_loader(split, shuffle=True):
			dataset	= MedTypeDataset(self.data[split], self.num_class, self.tokenizer, self.p)
			return DataLoader(
					dataset,
					batch_size      = self.p.batch_size * self.p.batch_factor,
					shuffle         = shuffle,
					num_workers     = self.p.num_workers,
					collate_fn      = dataset.collate_fn
				)

		self.data_iter = {
			'train'	: get_data_loader('train'),
			'valid'	: get_data_loader('valid', shuffle=False),
			'test'	: get_data_loader('test',  shuffle=False),
		}

	def add_model(self):
		"""
		Creates the computational graph
		Parameters
		----------
		
		Returns
		-------
		Creates the computational graph for model and initializes it
		
		"""
		if 	self.p.model == 'bert_plain': 		model = BertPlain(self.p, len(self.tokenizer), self.num_class)
		elif 	self.p.model == 'bert_combined': 	model = BertCombined(self.p, len(self.tokenizer), self.num_class)
		else:	raise NotImplementedError

		model = model.to(self.device)

		if len(self.gpu_list) > 1:
			print ('Using multiple GPUs ', self.p.gpu)
			model = nn.DataParallel(model, device_ids = list(range(len(self.p.gpu.split(',')))))
			torch.backends.cudnn.benchmark = True

		return model

	def add_optimizer(self, model, train_dataset_length):
		"""
		Creates an optimizer for training the parameters
		Parameters
		----------
		parameters:         The parameters of the model
		
		Returns
		-------
		Returns an optimizer and scheduler for learning the parameters of the model
		
		"""
		warmup_proportion 	= 0.1
		n_train_steps		= int(train_dataset_length / self.p.batch_size ) * self.p.max_epochs
		num_warmup_steps	= int(float(warmup_proportion) * float(n_train_steps))
		param_optimizer		= list(model.named_parameters())

		# Keeping bert params fixed for bert_combined model
		if self.p.model == 'bert_combined':
			param_optimizer = [x for x in param_optimizer if 'bert' not in x[0]]

		param_optimizer	= [n for n in param_optimizer if 'pooler' not in n[0]]
		no_decay	= ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]

		optimizer = AdamW(optimizer_grouped_parameters, lr=self.p.lr)
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=n_train_steps)
		return optimizer, scheduler

	def __init__(self, params):
		"""
		Constructor of the runner class
		Parameters
		----------
		params:         List of hyper-parameters of the model
		
		Returns
		-------
		Creates computational graph and optimizer
		
		"""
		self.p = params

		if not os.path.exists(self.p.log_dir):   os.system('mkdir -p {}'.format(self.p.log_dir))		# Create log directory if doesn't exist
		if not os.path.exists(self.p.model_dir): os.system('mkdir -p {}'.format(self.p.model_dir))		# Create model directory if doesn't exist

		# Get Logger
		self.logger	= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p)); pprint(vars(self.p))

		self.gpu_list = self.p.gpu.split(',')
		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model			= self.add_model()
		self.optimizer,self.scheduler	= self.add_optimizer(self.model, len(self.data['train']))

	def load_model(self, load_path):
		"""
		Function to load a saved model
		Parameters
		----------
		load_path: path to the saved model
		
		Returns
		-------
		"""
		state = torch.load('{}/{}'.format(load_path, self.p.name))
		self.best_val		= 0.0
		self.best_test		= 0.0
		self.best_epoch		= 0

		if len(self.gpu_list) > 1:
			state_dict 	= state['state_dict']
			new_state_dict  = OrderedDict()

			for k, v in state_dict.items():
				if 'module' not in k: 	k = 'module.' + k
				else: 			k = k.replace('features.module.', 'module.features.')
				new_state_dict[k] = v

			self.model.load_state_dict(new_state_dict)
		else:
			state_dict 	= state['state_dict']
			new_state_dict  = OrderedDict()

			for k, v in state_dict.items():
				if 'module' in k:
					k = k.replace('module.', '')

				new_state_dict[k] = v

			self.model.load_state_dict(new_state_dict)

		if self.p.restore_opt:
			self.optimizer.load_state_dict(state['optimizer'])
			self.best_test	= state['best_test']
			self.best_val	= state['best_val']
			self.best_epoch	= state['best_epoch']

	def save_model(self, save_path):
		"""
		Function to save a model. It saves the model parameters, best validation scores,
		best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
		Parameters
		----------
		save_path: path where the model is saved
		
		Returns
		-------
		"""
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_test'	: self.best_test,
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, '{}/{}'.format(save_path, self.p.name))

	def evaluate(self, logits, labels):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		logits: Predictions by the model
		labels: Ground truth labels
		
		Returns
		-------
		Area under PR-curve
		"""
		all_logits = np.concatenate(logits, axis=0)
		all_labels = np.concatenate(labels, axis=0)
		result = np.round(average_precision_score(all_labels.reshape(-1), all_logits.reshape(-1)), 3)
		return result

	def execute(self, batch):
		batch		= to_gpu(batch, self.device)
		loss, logits 	= self.model(
					input_ids	= batch['tok_pad'], 
					attention_mask	= batch['tok_mask'], 
					mention_pos_idx	= batch['men_pos'],
					labels		= batch['labels']
				)

		if len(self.gpu_list) > 1:
			loss = loss.mean()

		return loss, logits

	def predict(self, epoch, split, return_extra=False):
		"""
		Function 

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		
		Returns
		-------
		Loss and performance on the split
		"""
		self.model.eval()

		all_eval_loss, all_logits, all_labels, all_rest, cnt = [], [], [], [], 0

		with torch.no_grad():
			for batches in self.data_iter[split]:
				for k, batch in enumerate(batches):
					eval_loss, logits = self.execute(batch)

					if (k+1) % self.p.log_freq == 0:
						eval_res = self.evaluate(all_logits, all_labels)
						self.logger.info('[E: {}] | {:.3}% | {} | Eval {} --> Loss: {:.3}, Eval Acc: {}'.format(epoch, \
							100*cnt/len(self.data[split]),  self.p.name, split, np.mean(all_eval_loss), eval_res))

					all_eval_loss.append(eval_loss.item())
					all_logits.append(logits.cpu().numpy())
					all_labels.append(batch['labels'].cpu().numpy())

					if return_extra: all_rest.append(batch['_rest'])

					cnt += batch['tok_len'].shape[0]

		eval_res = self.evaluate(all_logits, all_labels)

		if return_extra: return np.mean(all_eval_loss), eval_res, all_logits, all_labels, all_rest
		else: 		 return np.mean(all_eval_loss), eval_res

	def check_and_save(self, epoch):
		valid_loss, valid_acc = self.predict(epoch, 'valid')

		if valid_acc > self.best_val:
			self.best_val		= valid_acc
			_, self.best_test	= self.predict(epoch, 'test')
			self.best_epoch		= epoch
			self.save_model(self.p.model_dir)
			return True
	
		return False


	def run_epoch(self, epoch, shuffle=True):
		"""
		Function to run one epoch of training
		Parameters
		----------
		epoch: current epoch count
		
		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		
		self.model.train()

		all_train_loss, all_score, cnt = [], [], 0

		for batches in self.data_iter['train']:
			for k, batch in enumerate(batches):
				self.optimizer.zero_grad()

				train_loss, logits = self.execute(batch)

				if (k+1) % self.p.log_freq == 0:
					eval_res = np.round(np.mean(all_score), 3)

					self.logger.info('[E: {}] | {:.3}% | {} | L: {:.3}, T: {}, B-V:{}'.format(epoch, \
						100*cnt/len(self.data['train']), self.p.name, np.mean(all_train_loss), eval_res, self.best_val))


				all_train_loss.append(train_loss.item())
				all_score.append(self.evaluate([logits.detach().cpu().numpy()], [batch['labels'].cpu().numpy()]))

				train_loss.backward()
				self.optimizer.step()
				self.scheduler.step()

				cnt += batch['tok_len'].shape[0]
				
		eval_res = np.round(np.mean(all_score), 3)

		return np.mean(all_train_loss), eval_res


	def fit(self):
		"""
		Function to run training and evaluation of model
		Parameters
		----------
		
		Returns
		-------
		"""

		self.best_val, self.best_test, self.best_epoch = 0.0, 0.0, 0

		if self.p.restore:
			self.load_model(self.p.model_dir)

			if self.p.dump_only:
				all_logits, all_labels, all_rest = [], [], []

				for split in ['test', 'valid']:
					loss, acc, logits, labels, rest = self.predict(0, split, return_extra=True)
					print('Score: Loss: {}, Acc:{}'.format(loss, acc))

					all_logits	+= logits
					all_labels	+= labels
					all_rest	+= rest

				dump_dir = './predictions/{}'.format(self.p.data); make_dir(dump_dir)
				dump_pickle({
					'logits': all_logits,
					'labels': all_labels,
					'others': all_rest
				}, '{}/{}'.format(dump_dir, self.p.name))

				exit(0)

		early_stop = 0
		for epoch in range(self.p.max_epochs):
			train_loss, train_acc	= self.run_epoch(epoch)

			if self.check_and_save(epoch): 
				early_stop = 0
			else:
				early_stop += 1
				if early_stop > self.p.early_stop:
					self.logger.info('Early Stopping!')
					break

			self.logger.info('Train loss: {:3}, Valid Perf: {:.3}'.format(train_loss, self.best_val))

		self.logger.info('Best Performance: {}'.format(self.best_test)) 

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='MedType Model Trainer')

	parser.add_argument('--gpu',      	default='0',                				help='GPU to use')
	parser.add_argument("--model", 		default='bert_plain', 	type=str, 			help='Type of model architecture. Options: `bert_plain` and `bert_combined`')

	# Model Specific
	parser.add_argument('--max_seq_len', 	default=128, 		type=int, 			help='Max allowed length of utt')
	parser.add_argument('--bert_model', 	default='bert-base-cased', 		type=str, 	help='Which Bert model')
	parser.add_argument('--data', 	 	default='medmentions', 			type=str, 	help='Which data')
	parser.add_argument('--model_wiki', 	default=None, 				type=str, 	help='Application when model == bert_combined | BERT model trained on WikiMed ')
	parser.add_argument('--model_pubmed', 	default=None, 				type=str, 	help='Application when model == bert_combined | BERT model trained on PubMedDS')

	parser.add_argument('--early_stop',    	dest='early_stop',	default=5,    	type=int,       help='Early Stop Count')
	parser.add_argument('--epoch',    	dest='max_epochs',	default=100,    type=int,       help='Max epochs')
	parser.add_argument('--batch',    	dest='batch_size',	default=16,     type=int,      	help='Batch size')
	parser.add_argument('--batch_factor',   dest='batch_factor',	default=50,     type=int,      	help='Number of batches to generate at one time')
	parser.add_argument('--num_workers',	type=int,		default=2,                   	help='Number of cores used for preprocessing data')
	parser.add_argument('--lr', 	 	default=1e-3, 		type=float, 			help='The initial learning rate for Adam.')
	parser.add_argument('--l2', 	 	default=0.0, 		type=float, 			help='The initial learning rate for Adam.')
	parser.add_argument('--drop', 	 	default=0.1, 		type=float, 			help='The initial learning rate for Adam.')

	parser.add_argument('--seed',     	default=1234,   	type=int,       		help='Seed for randomization')
	parser.add_argument('--log_freq',    	default=10,   		type=int,     			help='Display performance after these number of batches')
	parser.add_argument('--name',     	default='test',             				help='Name of the run')
	parser.add_argument('--restore',  				action='store_true',        	help='Restore from the previous best saved model')
	parser.add_argument('--restore_opt',  				action='store_true',        	help='Restore Optimizer from the previous best saved model')
	parser.add_argument('--dump_only',  				action='store_true',        	help='Dumps predictions for Entity Linking')

	parser.add_argument('--config_dir',   	default='../config',        				help='Config directory')
	parser.add_argument('--data_dir',   	default='./data',        				help='Config directory')
	parser.add_argument('--model_dir',   	default='./models',        				help='Model directory')
	parser.add_argument('--log_dir',   	default='./logs',   	   				help='Log directory')

	args = parser.parse_args()
	set_gpu(args.gpu)

	# Set seed
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Create Model
	model = MedType(args)
	model.fit()
	print('Model Trained Successfully!!')