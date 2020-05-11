DEFAULT='\033[0m'
BOLD='\033[1;32m\e[1m'

echo -e "${BOLD} MedType> Creating Directories ${DEFAULT}"

check_dir () { 
	if [ ! -d $1 ] 
	then 
		mkdir $1
	fi
}

check_dir "./data" 			# For data
check_dir "./logs" 			# For logs
check_dir "./models" 			# For storing model parameters
check_dir "./models/bert_dumps" 	# For storing fine-tuned BERT parameters
check_dir "./predictions" 		# For storing model's predictions
check_dir "./results" 			# For entity linking results

if [ ! -d "./models/bert_dumps/bert-base-cased" ]
then
	echo -e "${BOLD} MedType> Downloading BERT-base-cased ${DEFAULT}"
	wget https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/cased_L-12_H-768_A-12.tar.gz -O ./models/bert_dumps/bert-base-cased.tar.gz

	echo -e "${BOLD} MedType> Extracting BERT-base-cased ${DEFAULT}"
	tar -xzf ./models/bert_dumps/bert-base-cased.tar.gz -C ./models/bert_dumps/
	mv models/bert_dumps/cased_L-12_H-768_A-12 models/bert_dumps/bert-base-cased
	rm -f ./models/bert_dumps/bert-base-cased.tar.gz
fi 

if [ ! -f "./data/medmentions.pkl" ]
then
	echo -e "${BOLD} MedType> Downloading Processed MedMentions Data ${DEFAULT}"
	gdown --id 13Qszq-WGo8Ej9XTa7ccQ83_AdRZZ1vIU -O data/medmentions_processed.zip

	echo -e "${BOLD} MedType> Extracting MedMentions Data ${DEFAULT}"
	unzip data/medmentions_processed.zip -d data/
	rm -f data/medmentions_processed.zip
fi 

if [ ! -f "./data/ncbi.pkl" ]
then
	echo -e "${BOLD} MedType> Downloading Processed NCBI Data ${DEFAULT}"
	gdown --id 1NZkoIougiqPVXcxLCG_aT8sKR3jITUnI -O data/ncbi_processed.zip

	echo -e "${BOLD} MedType> Extracting NCBI Data ${DEFAULT}"
	unzip data/ncbi_processed.zip -d data/
	rm -f data/ncbi_processed.zip
fi


if [ ! -f "./data/umls2type.pkl" ]
then
	echo -e "${BOLD} MedType> Downloading UMLS to Semantic Type Mapping ${DEFAULT}"
	gdown --id 1Jly06IjI7iWgQRmj456filYD5FyBLQAg -O data/umls2type.zip
	unzip data/umls2type.zip -d data/
	rm -f data/umls2type.zip
fi 

if [ ! -f "./data/wikimed.pkl" ]
then
	echo -e "${BOLD} MedType> Downloading Processed WikiMed Data ${DEFAULT}"
	gdown --id 16tqtTVe74rTfHLjZU2lTTARNXT_y9MON -O data/wikimed_processed.zip

	echo -e "${BOLD} MedType> Extracting WikiMed Data ${DEFAULT}"
	unzip data/wikimed_processed.zip -d data/
	rm -f data/wikimed_processed.zip
fi

if [ ! -d "./data/pubmed_processed" ]
then
	echo -e "${BOLD} MedType> Downloading Processed PubMedDS Data ${DEFAULT}"
	gdown --id 1CPQnL-Ik1yCeCE8uccP_OmRSEGo4FkDB -O data/pubmed_processed.zip

	echo -e "${BOLD} MedType> Extracting PubMedDS Data ${DEFAULT}"
	unzip data/pubmed_processed.zip -d data/
	rm -f data/pubmed_processed.zip
fi


if [ ! -d "./models/pretrained_models" ]
then
	echo -e "${BOLD} MedType> Setting up directories ${DEFAULT}"
	mkdir -p models/pretrained_models
fi

if [ ! -f "./models/pretrained_models/general_model.bin" ]
then
	# Pretrained model for General domain text
	echo -e "${BOLD} MedType> Downloading pre-trained model for general articles: general_model.bin ${DEFAULT}"
	gdown --id 1lFFvd7XT9P1ZA_s7NDjUSm2YBRmo2TzM -O models/pretrained_models/general_model.zip

	echo -e "${BOLD} MedType> Extracting model ${DEFAULT}"
	unzip models/pretrained_models/general_model.zip -d models/pretrained_models/
	rm -f models/pretrained_models/general_model.zip
fi

if [ ! -f "./models/pretrained_models/pubmed_model.bin" ]
then
	# Pretrained model for Biomedical research articles
	echo -e "${BOLD} MedType> Downloading pre-trained model for biomedical articles: pubmed_model.bin ${DEFAULT}"
	gdown --id 19DrhHCpwOJX9aUBlDMDmyyQ3eSSdnEzO -O models/pretrained_models/pubmed_model.zip

	echo -e "${BOLD} MedType> Extracting model ${DEFAULT}"
	unzip models/pretrained_models/pubmed_model.zip -d models/pretrained_models/
	rm -f models/pretrained_models/pubmed_model.zip
fi


if [ ! -f "./models/pretrained_models/ehr_model.bin" ]
then
	# Pretrained model for EHR documents
	echo -e "${BOLD} MedType> Downloading pre-trained model for Electronic Health Records (EHRs): ehr_model.bin ${DEFAULT}"
	gdown --id 1Ft-yeC7af3MtypjejPmvnqcy7xRQuWbp -O models/pretrained_models/ehr_model.zip

	echo -e "${BOLD} MedType> Extracting model ${DEFAULT}"
	unzip models/pretrained_models/ehr_model.zip -d models/pretrained_models/
	rm -f models/pretrained_models/ehr_model.zip
fi

echo -e "${BOLD} MedType> All Set! ${DEFAULT}"