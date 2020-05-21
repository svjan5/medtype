from helper import *

from torch.nn import functional as F
from transformers import BertModel

class BertPlain(nn.Module):
	def __init__(self, params, num_labels):
		super().__init__()
		
		self.p 		= params
		self.bert 	= BertModel.from_pretrained(self.p.bert_model)
		self.dropout	= nn.Dropout(self.p.drop)
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

		if labels is None:
			return logits
		else:
			loss = F.binary_cross_entropy_with_logits(logits, labels.float())
			return loss, logits

class BertCombined(nn.Module):

	def __init__(self, params, num_labels):
		super().__init__()

		self.p			= params
		self.bert_wiki		= BertModel.from_pretrained('{}/bert_dumps/{}'.format(self.p.model_dir, self.p.wiki_model))
		self.bert_pubmed	= BertModel.from_pretrained('{}/bert_dumps/{}'.format(self.p.model_dir, self.p.pubmed_model))
		self.dropout		= nn.Dropout(self.p.drop)

		class_in	= self.bert_wiki.config.hidden_size + self.bert_pubmed.config.hidden_size
		self.classifier	= nn.Linear(class_in, num_labels)

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
		out_wiki = self.bert_wiki(
			input_ids 	= input_ids,
			attention_mask	= attention_mask,
			token_type_ids	= token_type_ids,
			position_ids	= position_ids,
			head_mask	= head_mask,
			inputs_embeds	= inputs_embeds,
		)

		out_pubmed = self.bert_pubmed(
			input_ids 	= input_ids,
			attention_mask	= attention_mask,
			token_type_ids	= token_type_ids,
			position_ids	= position_ids,
			head_mask	= head_mask,
			inputs_embeds	= inputs_embeds,
		)

		pooled_output	= torch.cat([out_wiki[1], out_pubmed[1]], dim=1)
		pooled_output	= self.dropout(pooled_output)
		logits		= self.classifier(pooled_output)

		if labels is None:
			return logits
		else:
			loss = F.binary_cross_entropy_with_logits(logits, labels.float())
			return loss, logits