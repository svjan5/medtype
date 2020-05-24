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
  medtype-trainer
</h1>

### Dependencies

- Compatible with PyTorch 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.
- Run `setup.sh` to setting up the environment and downloading the processed datasets. 

### Usage:

**MedType** can be trained from scratch using `python medtype.py`. The details about different arguments is provided below:

|   Argument    | Type  |     Default     | Description                                                  |
| :-----------: | :---: | :-------------: | :----------------------------------------------------------- |
|    `model`    |  str  |   bert_plain    | Type of model architecture. Options: `bert_plain` and `bert_combined` |
|     `gpu`     |  str  |        0        | List of GPUs to use (can use multiple). Example: `0`, `0,1,2,3`... |
| `max_seq_len` |  int  |       128       |                                                              |
| `bert_model`  |  str  | bert-base-cased | Used for initializing BERT in the architecture. Can be any model from [HuggingFace-Transformers](<https://huggingface.co/models>) |
|    `data`     |  str  |   medmentions   | Dataset to train and evaluate on.                            |
|    `epoch`    |  int  |       100       | Maximum number of epochs                                     |
| `early_stop`  |  int  |        5        | Early stop count. If valid performance doesn't improve for consecutive `early_stop` epochs then training is stopped. |
|     `lr`      | float |      0.001      | Learning rate                                                |
|     `l2`      | float |       0.0       | L2 regularization                                            |
|    `drop`     | float |       0.1       | Dropout to be applied on BERT output                         |
|    `seed`     |  int  |      1234       | Seed                                                         |
|    `name`     |  str  |      test       | Name of the model. Used for storing model's parameters.      |
|   `restore`   | bool  |      False      | To restore model's parameter                                 |
| `restore_opt` | bool  |      False      | To restore model's optimizer                                 |
|  `log_freq`   |  int  |       10        | Writes to log and stdout after `log_freq` batches            |
| `config_dir`  |  str  |    ../config    | Config directory                                             |
|  `data_dir`   |  str  |     ./data      | Data directory                                               |
|  `model_dir`  |  str  |    ./models     | Directory where trained model's parameters are stored        |
|   `log_dir`   |  str  |     ./logs      | Log directory                                                |


### Reproducing Best Results:

Below, we list the steps for reproducing the results for the best reported model (`T <- Both`, refer to the [paper](<https://arxiv.org/abs/2005.00460>))

1. Download the processed versions of the datasets and other directory structures required for running the code. 

   ```
   ./setup.sh
   ```

2. Train a model on `WikiMed` dataset. `BERT` parameters are initialized using `bert-base-cased` for keeping it more efficient at handling general text. 

   ```shell
   python medtype.py --data wikimed --name wiki_only --bert_model bert-base-cased
   ```

3. Train a model on `PubMedDS` dataset. We initialize this model's parameters with BioBERT for making it more efficient at handling biomedical research domain text. 

   ```shell
   python medtype.py --data pubmed_ds --name pubmed_only --bert_model monologg/biobert_v1.1_pubmed
   ```

4. Fine-tune both the models on training split of a dataset, say `MedMentions`

   ```shell
   # Fine-tune model trained in WikiMed
   cp models/wiki_only models/wiki_with_medmentions
   python medtype.py --data medmentions --name wiki_with_medmentions --restore
   
   # Fine-tune model trained in PubMedDS
   cp models/pubmed_only models/pubmed_with_medmentions
   python medtype.py --data medmentions --name pubmed_with_medmentions --restore
   ```

5. Dump `BERT` weights of the fine-tuned models separately for training a combined model in the next step.

   ```shell
   python dump_bert_weights.py --model wiki_with_medmentions
   python dump_bert_weights.py --model pubmed_with_medmentions
   ```

6. Finally, trained the `BertCombined` model using the weights trained on both `WikiMed` and `PubMedDS`. This gives the best performance overall. 

   ```shell
   python medtype.py --data medmentions \
   				  --model bert_combined \
   				  --name both_with_medmentions \
   				  --wiki_model wiki_with_mentions \
   				  --pubmed_model pubmed_with_medmentions
   ```

7. Path to `both_with_medmentions` model can be used as an input to **medtype-as-service**. Since, the `MedMentions` dataset comprises of annotated PubMed abstracts, therefore, most likely it will give the best performance on input from a similar domain.


For training a model for general text corpus, just use the model trained on `WikiMed` dataset. Similarly, one can obtain a model for handling Electronic Health Records (EHR) by training on annotated EHRs. 

### Pre-trained models

We provide three pre-trained models for tackling different domain:

- [**General text**](https://drive.google.com/file/d/1OJ66mvs5yw_EcOEaVEvABzMAWRAsoqR9/view?usp=sharing) (trained on WikiMed)
- [**Bio-Medical Research Articles**](https://drive.google.com/file/d/1XuFx5_q_6PCYQXNHb50DBc3PhJn2Gy1D/view?usp=sharing) (trained on WikiMed+PubMedDS+Annotated PubMed abstracts)
- [**Electronic Health Records (EHR)**](https://drive.google.com/file/d/1OJ66mvs5yw_EcOEaVEvABzMAWRAsoqR9/view?usp=sharing) (trained on WikiMed+PubMedDS+Annotated EHR documents)

### Evaluation

Since MedType works on the output of an entity linker, we need to first dump its output on a given dataset using:

```shell
python dump_linkers_output.py --model scispacy --data medmentions
```

We use [neleval](<https://github.com/wikilinks/neleval>) for evaluation which requires us to dump the predictions in a required format. The following code does that work for us and giving us the final results on entity linking.

```shell
# Base performance of entity linker
python eval_models.py --model scispacy --data medmentions

# Performance with Oracle (With entity disambiguation)
python eval_models.py --model scispacy --data medmentions --ent_disamb
```

Finally, for getting performance with **MedType**, we need to first dump the predictions of the trained model and finally run `eval_models.py`

```shell
# Dump the predictions of model on MedMentions
python medtype.py --data medmentions \
				  --model bert_combined \
				  --name both_with_medmentions \
				  --wiki_model wiki_with_mentions \
				  --pubmed_model pubmed_with_medmentions \
				  --dump_only --restore 
				  
# Performance with MedType predictions (With entity disambiguation)
python eval_models.py --model scispacy --data medmentions --ent_disamb --pred_model both_with_medmentions
```

