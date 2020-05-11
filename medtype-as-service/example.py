from medtype_serving.client import MedTypeClient
from pprint import pprint

client = MedTypeClient(ip='localhost')
text   = ['Symptoms of common cold includes cough, fever, high temperature and nausea.']
pprint(client.run_linker(text)['elinks'])