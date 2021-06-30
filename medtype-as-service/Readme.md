<h1 align="center">
  <img align="center" width="450" src="../images/logo.png" alt="...">
</h1>



<h4 align="center">Improving Medical Entity Linking with Semantic Type Prediction</h4>

<p align="center">
  <a href="https://arxiv.org/abs/2005.00460"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <a href="https://medtype.github.io"><img src="http://img.shields.io/badge/Demo-Live-green.svg"></a>
  <a href="https://github.com/svjan5/medtype/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>


<h1 align="center">
  medtype-as-service
</h1>


### Dependencies

- **Server:** Compatible with PyTorch 1.x and Python 3.x.
- **Client:** Both Python 2.x and 3.x.
- Individual dependencies for client and server installed from `requirements.txt`.
- Run `./setup.sh` for downloading the pre-trained models and other essential files.

### Setting up Entity Linkers:

1. **ScispaCy:**  Follow the instructions given [here](<https://github.com/allenai/scispacy>). In short just run:

   ```shell
   pip install scispacy
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
   ```

2. **QuickUMLS:** Requires installing a copy of UMLS. Please follow instructions given [here](https://github.com/Georgetown-IR-Lab/QuickUMLS).

3. **MetaMap & MetaMapLite**: Follow the steps given [here](<https://github.com/AnthonyMRios/pymetamap>). However, for the sake of simplicity we summarize the steps below:

   1. Download Metamap ([model](<https://metamap.nlm.nih.gov/MainDownload.shtml>)) and MetamapLite ([model](https://metamap.nlm.nih.gov/download/metamaplite/public_mm_lite_3.6.2rc3_binaryonly.zip)+[data](https://metamap.nlm.nih.gov/download/metamaplite/public_mm_data_lite_usabase_2018ab_ascii.zip)) files. Please note it requires credential  access to UMLS to download these files.

   2. Extract all the above downloaded files. Then, do

      ```shell
      # For MeteMap
      cd public_mm/bin; 
      ./install.sh
      ./bin/skrmedpostctl start
      ./bin/wsdserverctl start
      
      # For MeteMapLite
      cd public_mm_lite
      mv data/ivf/2018ABascii data/ivf/2018AB
      mkdir data/ivf/2018AB/USAbase/strict
      mv data/ivf/2018AB/USAbase/* data/ivf/2018AB/USAbase/strict
      ```

   3. Finally, check whether everything is installed perfectly by running the following:

      ```shell
      git clone https://github.com/AnthonyMRios/pymetamap
      cd pymetamap; 
      python setup.py install
      
      python
      >>> from pymetamap import MetaMap, MetaMapLite
      >>> sents = ['Heart Attack', 'John had a huge heart attack']
      >>> metamap = MetaMap.get_instance('<give absolute path>/public_mm/bin/metamap18')
      >>> metamap.extract_concepts(sents,[1,2])
      
      >>> metamaplite = MetaMapLite.get_instance('<give absolute path>/public_mm_lite/')
      >>> metamaplite.extract_concepts(sents,[1,2])
      ```

4. **CTakes**: Download the CTakes directory and resources to $CTAKES_HOME. Download and set up the ctakes-server given [here](<https://github.com/dirkweissenborn/ctakes-server>). We summarize the steps below:

   1. Download [cTakes software](https://downloads.apache.org//ctakes/ctakes-4.0.0/apache-ctakes-4.0.0-bin.tar.gz) and copy the [resources](<https://managedway.dl.sourceforge.net/project/ctakesresources/ctakes-resources-4.0-bin.zip>) in `resources`  directory as given below:

      ```shell
      wget https://downloads.apache.org//ctakes/ctakes-4.0.0/apache-ctakes-4.0.0-bin.tar.gz
      wget https://managedway.dl.sourceforge.net/project/ctakesresources/ctakes-resources-4.0-bin.zip
      
      tar -xzf apache-ctakes-4.0.0-bin.tar.gz
      unzip ctakes-resources-4.0-bin.zip
      
      rsync -av resources/ apache-ctakes-4.0.0/resources/
      
      export CTAKES_HOME=$PWD/apache-ctakes-4.0.0
      ```

   2. Now install ctakes-server repository and setup `resources` and `desc` directories:

      ```
      git clone github.com/dirkweissenborn/ctakes-server
      cd ctakes-server
      
      ln -s $CTAKES_HOME/resources resources
      ln -s $CTAKES_HOME/desc desc
      ```

   3. Change to your CTakes version in pom.xml to `4.0.0`. After that run `mvn_package`. Please note that Java version should be 1.8.

   4. Change DictionaryLookupAnnotatorDB to the following `(In file : dec/ctakes-clinical-pipeline/desc/analysis_engine/AggregatePlaintextUMLSProcessor.xml)`

      ```xml
      <delegateAnalysisEngine key="DictionaryLookupAnnotatorDB"> 
      	<!--import location="../../../ctakes-dictionary-lookup/desc/analysis_engine/DictionaryLookupAnnotatorUMLS.xml" /--> 
      	<import location="../../../ctakes-dictionary-lookup-fast/desc/analysis_engine/UmlsLookupAnnotator.xml"/> 
      </delegateAnalysisEngine>
      ```

   5. To start CTakes server, use : 

      ```shell
      java -Dctakes.umlsuser=<XXXX> \
      	 -Dctakes.umlspw=<XXXX> \
      	 -Xmx5g -cp target/ctakes-server-0.1.jar:resources/ \
      	 de.dfki.lt.ctakes.Server localhost 9999 \
      	 desc/ctakes-clinical-pipeline/desc/analysis_engine/AggregatePlaintextUMLSProcessor.xml
      ```

      `localhost` and `9999` are ip and port of the ctakes-server. New authentication instructions for ctakes: [link](https://issues.apache.org/jira/secure/attachment/13015458/Readme_for_UMLS_auth.txt).

   6. To check whether everything is proper installed query : 

      ```shell
      curl "http://localhost:9999/ctakes?text=Pain in the left leg." 
      ```

### Pre-trained Models for tackling different domain:

- [**General text**](https://drive.google.com/file/d/15vKHwzEa_jcipyEDClNSzJguPxk0VOC7/view?usp=sharing) (trained on WikiMed)
- [**Bio-Medical Research Articles**](https://drive.google.com/file/d/1So-FMFyPMup84VvbWqH7Cars8jfjEIx_/view?usp=sharing) (trained on WikiMed+PubMedDS+Annotated PubMed abstracts)
- [**Electronic Health Records (EHR)**](https://drive.google.com/file/d/1t2QlpEWnHOMdts4h3y55hVA9Wh2ZbjKi/view?usp=sharing) (trained on WikiMed+PubMedDS+Annotated EHR documents)

### Usage

Run `./setup.sh` for installing **medtype-as-service** server and cilent packages and downloading other resources for running the service.  

For staring server:

```shell
medtype-serving-start --model_path $PWD/resources/pretrained_models/pubmed_model.bin \
		      --type_remap_json $PWD/../config/type_remap.json \
		      --type2id_json $PWD/../config/type2id.json \
		      --umls2type_file $PWD/resources/umls2type.pkl \ 
		      --entity_linker scispacy

# Debug mode: (In case one doesn't want to reinstall the package after every change)
cd server
python -c 'import medtype_serving.server.cli as cli; cli.main()'  \
			  --model_path $PWD/../resources/pretrained_models/pubmed_model.bin \
		      --type_remap_json $PWD/../../config/type_remap.json \
		      --type2id_json $PWD/../../config/type2id.json \
		      --umls2type_file $PWD/../resources/umls2type.pkl \ 
		      --entity_linker scispacy
```

On client side:

```python
from medtype_serving.client import MedTypeClient
from pprint import pprint

client  = MedTypeClient(ip='localhost')
message = {
	'text': ['Symptoms of common cold includes cough, fever, high temperature and nausea.'],
	'entity_linker': 'scispacy'
}

pprint(client.run_linker(message)['elinks'])
```

### Running HTTP-based Server:

**On server side:**

```shell
medtype-serving-start --model_path $PWD/resources/pretrained_models/pubmed_model.bin \
		      --type_remap_json $PWD/../config/type_remap.json \
		      --type2id_json $PWD/../config/type2id.json \
		      --umls2type_file $PWD/resources/umls2type.pkl \ 
		      --entity_linker scispacy --http_port 8125
```

For serving over HTTPs, execute:

```shell
openssl genrsa 1024 > ssl.key
openssl req -new -x509 -nodes -sha1 -days 365 -key ssl.key > ssl.crt

medtype-serving-start --enable_https .....
```
The above script has been used for hosting our [online demo](http://medtype.github.io/).

**On client side:**

```python
curl -X POST http://xx.xx.xx.xx:8125/run_linker   -H 'content-type: application/json'   -d '{"id": 123,"data": {"text":["Pain in the left leg."], "entity_linker": "scispacy"}}'
```



### Server API

| Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_path` | str | *Required* | path of the pre-trained **MedType** model. |
| `type_remap_json`           | str | *Required* | Json file containing semantic type remapping to coarse-grained types |
| `type2id_json`           | str | *Required* | Json file containing semantic type to identifier mapping |
| `umls2type_file`           | str | *Required* | Location where UMLS to semantic types mapping is stored |
| `entity_linker`       | str   | scispacy        | entity linker to use over which MedType is employed. Options: [scispacy, quickumls, ctakes, metamap, metamaplite]. Can load multiple entity linkers as well by concatenating them with `,`. |
| `model_type` | str | `bert_combined` | Model type. Options: [bert_combined (Bio arcticles/EHR), bert_plain (General)] |
| `tokenizer_model`     | str   | `bert-base-cased` | Tokenizer used for chunking text.                            |
| `context_len`         | int   | `120`           | Number of neighboring tokens to consider for predicting semantic type of a mention |
| `threshold`           | float | `0.5`           | Threshold used on logits from MedType for getting semantic types |
| `quickumls_path`           | str | `0.5`           | Location where QuickUMLS is installed |
| `dropout`           | str | `0.1`           | Dropout in MedType Model |
| `max_seq_len` | int | `25` | maximum length of sequence, longer sequence will be trimmed on the right side. Set it to NONE for dynamically using the longest sequence in a (mini)batch. |
| `model_batch_size`    | int   | `256`           | maximum number of mentions handled by MedType model          |
| `num_worker` | int | `1` | number of (GPU/CPU) worker runs BERT model, each works in a separate process. |
| `max_batch_size` | int | `256` | maximum number of sequences handled by each worker, larger batch will be partitioned into small batches. |
| `priority_batch_size` | int | `16` | batch smaller than this size will be labeled as high priority, and jumps forward in the job queue to get result faster |
| `port` | int | `5555` | port for pushing data from client to server |
| `port_out` | int | `5556`| port for publishing results from server to client |
| `http_port` | int | None | server port for receiving HTTP requests |
| `enable_https` | bool | False | Enable serving HTTPs requests |
| `cors` | str | `*` | setting "Access-Control-Allow-Origin" for HTTP requests |
| `gpu_memory_fraction` | float | `0.5` | the fraction of the overall amount of memory that each GPU should be allocated per worker |
| `cpu` | bool | False | run on CPU instead of GPU |
| `device_map` | list | `[]` | specify the list of GPU device ids that will be used (id starts from 0)|

### Client API

Client-side provides a Python class called `MedTypeClient`, which accepts arguments as follows:

| Argument | Type | Default | Description |
|----------------------|------|-----------|-------------------------------------------------------------------------------|
| `ip` | str | `localhost` | IP address of the server |
| `port` | int | `5555` | port for pushing data from client to server, *must be consistent with the server side config* |
| `port_out` | int | `5556`| port for publishing results from server to client, *must be consistent with the server side config* |
| `show_server_config` | bool | `False` | whether to show server configs when first connected |
| `identity` | str | `None` | a UUID that identifies the client, useful in multi-casting |
| `timeout` | int | `-1` | set the timeout (milliseconds) for receive operation on the client |
