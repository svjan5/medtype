from medtype_serving.client import MedTypeClient
from pprint import pprint
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--linker', default='scispacy')
args = parser.parse_args()

client  = MedTypeClient(ip='localhost')
message = {
	'text': ['Symptoms of common cold includes cough, fever, high temperature and nausea.'],
	'entity_linker': args.linker
}

pprint(client.run_linker(message)['elinks'])