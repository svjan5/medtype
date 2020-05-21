import torch, torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig, BertPreTrainedModel

class BertPlain(nn.Module):
	def __init__(self, num_labels, dropout):
		super().__init__()
		
		self.bert 	= BertModel.from_pretrained('bert-base-cased')
		self.dropout	= nn.Dropout(dropout)
		self.classifier	= nn.Linear(self.bert.config.hidden_size, num_labels)
	
	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
		outputs = self.bert(
			input_ids 	= input_ids,
			attention_mask	= attention_mask,
			token_type_ids	= token_type_ids,
			position_ids	= position_ids,
			head_mask	= head_mask,
			inputs_embeds	= inputs_embeds
		)

		pooled_output	= outputs[1]
		pooled_output	= self.dropout(pooled_output)
		logits		= self.classifier(pooled_output)

		return logits

class BertCombined(nn.Module):

	def __init__(self, num_labels, dropout):
		super().__init__()
		
		self.bert_wiki		= BertModel.from_pretrained('bert-base-cased')
		self.bert_pubmed	= BertModel.from_pretrained('bert-base-cased')
		self.dropout		= nn.Dropout(dropout)

		class_in		= self.bert_wiki.config.hidden_size + self.bert_pubmed.config.hidden_size
		self.classifier		= nn.Linear(class_in, num_labels)

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
		out_wiki = self.bert_wiki(
			input_ids 	= input_ids,
			attention_mask	= attention_mask,
			token_type_ids	= token_type_ids,
			position_ids	= position_ids,
			head_mask	= head_mask,
			inputs_embeds	= inputs_embeds,
		)

		pooled_wiki = out_wiki[1]

		out_pubmed = self.bert_pubmed(
			input_ids 	= input_ids,
			attention_mask	= attention_mask,
			token_type_ids	= token_type_ids,
			position_ids	= position_ids,
			head_mask	= head_mask,
			inputs_embeds	= inputs_embeds,
		)

		pooled_pubmed   = out_pubmed[1]
		pooled_output   = torch.cat([pooled_wiki, pooled_pubmed], dim=1)
		pooled_output	= self.dropout(pooled_output)
		logits		= self.classifier(pooled_output)

		return logits
		
