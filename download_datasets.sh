DEFAULT='\033[0m'
BOLD='\033[1;32m\e[1m'

if ! python -c "import gdown" &> /dev/null; then
	echo -e "${BOLD} MedType> Install gdown package ${DEFAULT}"
	pip install gdown
fi

if [ ! -d "./datasets" ]
then
	echo -e "${BOLD} MedType> Setting up directories ${DEFAULT}"
	mkdir datasets
fi

if [ ! -f "./datasets/wikipedia2umls.json" ]
then
	# For Wikipedia to UMLS Mapping
	echo -e "${BOLD} MedType> Downloading Wikipedia to UMLS mapping ${DEFAULT}"
	gdown --id 1WjSEn2UNoYgpWcRI2Up2eRXIsnSvEnna -O datasets/wikipedia2umls.zip

	echo -e "${BOLD} MedType> Extracting Wikipedia to UMLS mapping ${DEFAULT}"
	unzip datasets/wikipedia2umls.zip -d datasets/
	rm -f datasets/wikipedia2umls.zip
fi

if [ ! -f "./datasets/medmentions.json" ]
then
	# For MedMentions
	echo -e "${BOLD} MedType> Downloading MedMentions dataset ${DEFAULT}"
	gdown --id 1E_cSs3GJy84oATsMBYE7xMEoif-f4Ei6 -O datasets/medmentions.zip

	echo -e "${BOLD} MedType> Extracting MedMentions dataset ${DEFAULT}"
	unzip datasets/medmentions.zip -d datasets/
	rm -f datasets/medmentions.zip
fi

if [ ! -f "./datasets/ncbi.json" ]
then
	# For NCBI Disease Corpus
	echo -e "${BOLD} MedType> Downloading NCBI dataset ${DEFAULT}"
	gdown --id 1SawFWcHgXSwQu-CA5tb46XCbNRIXo4Sf -O datasets/ncbi.zip

	echo -e "${BOLD} MedType> Extracting NCBI dataset ${DEFAULT}"
	unzip datasets/ncbi.zip -d datasets/
	rm -f datasets/ncbi.zip
fi


if [ ! -f "./datasets/wikimed.json" ]
then
	# For WikiMed
	echo -e "${BOLD} MedType> Downloading WikiMed dataset ${DEFAULT}"
	gdown --id 16suJCinjfYhw1u1S-gPFmGFQZD331u7I -O datasets/wikimed.zip

	echo -e "${BOLD} MedType> Extracting WikiMed dataset ${DEFAULT}"
	unzip datasets/wikimed.zip -d datasets/
	rm -f datasets/wikimed.zip
fi

if [ ! -d "./datasets/pubmed_ds" ]
then
	# For PubMedDS
	echo -e "${BOLD} MedType> Downloading PubMedDS dataset ${DEFAULT}"
	gdown --id 16mEFpCHhFGuQ7zYRAp2PP3XbAFq9MwoM -O datasets/pubmed_ds.zip

	echo -e "${BOLD} MedType> Extracting PubMedDS dataset ${DEFAULT}"
	unzip datasets/pubmed_ds.zip -d datasets/
	rm -f datasets/pubmed_ds.zip

fi

echo -e "${BOLD} MedType> All Set! ${DEFAULT}"