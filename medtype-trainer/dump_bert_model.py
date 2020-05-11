from helper import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default=None, required=True)
args = parser.parse_args()

base_dir = '{}/type_pred/models/bert_dumps/{}'.format(PROJ_DIR, args.model); make_dir(base_dir)
state    = torch.load('{}/type_pred/models/test/{}'.format(PROJ_DIR, args.model))
torch.save(state['state_dict'], '{}/pytorch_model.bin'.format(base_dir))
os.system('cp ./models/bert_dumps/bert-base-cased/config.json {}'.format(base_dir))
os.system('cp ./models/bert_dumps/bert-base-cased/vocab.txt {}'.format(base_dir))