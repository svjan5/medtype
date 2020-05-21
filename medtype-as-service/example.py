from medtype_serving.client import MedTypeClient
from pprint import pprint

client  = MedTypeClient(ip='localhost')
message = {
	'text': ['Symptoms of common cold includes cough, fever, high temperature and nausea.'],
	'entity_linker': 'scispacy'
}

pprint(client.run_linker(message)['elinks'])