from helper import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default=None, required=True)
args = parser.parse_args()

base_dir   = '{}/type_pred/models/bert_dumps/{}'.format(PROJ_DIR, args.model); make_dir(base_dir)
state	   = torch.load('{}/type_pred/models/test/{}'.format(PROJ_DIR, args.model))

new_state  = OrderedDict()
for key, val in state['state_dict'].items():
	new_state[key.replace('module.', '')] = val

torch.save(new_state, '{}/pytorch_model.bin'.format(base_dir))

os.system('cp ./models/bert_dumps/bert-base-cased/config.json {}'.format(base_dir))
os.system('cp ./models/bert_dumps/bert-base-cased/vocab.txt {}'.format(base_dir))