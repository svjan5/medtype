# A Dockerfile for running the MedType server.

# Use anaconda for prerequisites.
FROM conda/miniconda3

# Update conda
RUN conda update conda python

# Install unzip from conda.
RUN conda install -c conda-forge unzip

# Install gcc and g++
RUN conda install -c conda-forge gcc
RUN conda install -c conda-forge gxx

# Install rust
RUN conda install -c anaconda rust-nightly 

# Install prerequisites from server/requirements.txt
# Start with the largest ones so we don't have to redownload them.
RUN conda install -c pytorch pytorch

# Other prerequisites
RUN conda install -c anaconda numpy
RUN conda install -c conda-forge tokenizers
RUN conda install -c conda-forge spacy
RUN conda install -c anaconda nltk
RUN conda install -c anaconda flask
RUN conda install -c anaconda zeromq

# Install SciSpaCy as an entity linker.
RUN pip install scispacy
# RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

# We need pip to set up Python packages.
RUN pip install --upgrade pip

# Predownload some scispacy files.
RUN mkdir -p /root/.scispacy/datasets
ADD 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2020-10-09/umls/tfidf_vectors_sparse.npz' /root/.scispacy/datasets/e9f7327283e43f0482f7c0c71b71dec278a58ccb3ffdd03c2c2350159e7ef146.f2a350ad19015b2591545f7feeed6a6d6d2fffcd635d868a5d7fc0dfc3cadfd8.tfidf_vectors_sparse.npz
ADD 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2020-10-09/umls/nmslib_index.bin' /root/.scispacy/datasets/f48455d6c79262057cce66b4619123c2b558b21092d42fac97f47bb99a5b8f9f.dd70d3dffe7d90d7ac8914460e16a48375dab32485fb6313a34e6fbcaf53218b.nmslib_index.bin
ADD 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2020-10-09/umls/tfidf_vectorizer.joblib' /root/.scispacy/datasets/8c32f1e7ddf19ec695c321f68a71f06a191aec8efcf6b645b78fa6250d8d81d3.89019b4a62a096f33ea23677557a4cde66ebc8228f30afabac38e32f834020dc.tfidf_vectorizer.joblib
ADD 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2020-10-09/umls/concept_aliases.json' /root/.scispacy/datasets/1428ec15d3b1061731ea273c03699130b3d6b90948993e74bda66af605ff8e2a.aeb7a686c654df6bccb6c2c23d3eda3eb381daaefda4592b58158d0bee53b352.concept_aliases.json
ADD 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/kbs/2020-10-09/umls_2020_aa_cat0129.jsonl' /root/.scispacy/datasets/4d7fb8fcae1035d1e0a47d9072b43d5a628057d35497fbfb2499b4b7b2dd4dd7.05ec7eef12f336d4666da85b7fa69b9401883a7dd4244473f7b88b413ccbba03.umls_2020_aa_cat0129.jsonl
ADD 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/umls_semantic_type_tree.tsv' /root/.scispacy/datasets/21a1012c532c3a431d60895c509f5b4d45b0f8966c4178b892190a302b21836f.330707f4efe774134872b9f77f0e3208c1d30f50800b3b39a6b8ec21d9adf1b7.umls_semantic_type_tree.tsv

# Copy the source code to /opt/medtype-as-service
WORKDIR /opt/medtype-as-service
COPY . /opt/medtype-as-service

# Install MedType Server requirements
# RUN pip install -r server/requirements.txt

# Setup MedType as service
RUN bash setup.sh

# Run MedType server
EXPOSE 8125/tcp
ENTRYPOINT medtype-serving-start --model_path $PWD/resources/pretrained_models/pubmed_model.bin \
  --type_remap_json $PWD/config/type_remap.json \
  --type2id_json $PWD/config/type2id.json \
  --umls2type_file $PWD/resources/umls2type.pkl \ 
  --entity_linker scispacy \
  --http_port 8125
