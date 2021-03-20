from helper import *

from torch.nn import functional as F
from transformers import BertModel

class BertPlain(nn.Module):
	def __init__(self, params, num_tokens, num_labels):
		super().__init__()
		
		self.p 		= params
		self.bert 	= BertModel.from_pretrained(self.p.bert_model)
		self.bert.resize_token_embeddings(num_tokens)
		self.dropout	= nn.Dropout(self.p.drop)
		self.classifier	= nn.Linear(self.bert.config.hidden_size, num_labels)
	
	def forward(self, input_ids, attention_mask, mention_pos_idx, labels=None):
		outputs = self.bert(
			input_ids 	= input_ids,
			attention_mask	= attention_mask
		)

		tok_embed	= outputs[0]
		bsz, mtok, dim  = tok_embed.shape
		tok_embed_flat	= tok_embed.reshape(-1, dim)
		men_idx 	= torch.arange(bsz).to(tok_embed.device) * mtok + mention_pos_idx
		men_embed 	= tok_embed_flat[men_idx]

		pooled_output	= self.dropout(men_embed)
		logits		= self.classifier(pooled_output)
		loss 		= F.binary_cross_entropy_with_logits(logits, labels.float())

		loss = F.binary_cross_entropy_with_logits(logits, labels.float())
		return loss, logits

class BertCombined(nn.Module):

	def __init__(self, params, num_tokens, num_labels):
		super().__init__()

		self.p			= params
		self.bert_wiki		= BertModel.from_pretrained('models/bert_dumps/{}'.format(self.p.wiki_model)); 	
		self.bert_pubmed	= BertModel.from_pretrained('models/bert_dumps/{}'.format(self.p.pubmed_model));
		self.dropout		= nn.Dropout(self.p.drop)

		if self.p.comb_opn == 'concat': class_in = self.bert_wiki.config.hidden_size * 2
		else: 				class_in = self.bert_wiki.config.hidden_size

		self.classifier	 = nn.Linear(class_in, num_labels)

	def forward(self, input_ids, attention_mask, mention_pos_idx, labels=None):
		out_wiki = self.bert_wiki(
			input_ids 	= input_ids,
			attention_mask	= attention_mask
		)

		out_pubmed = self.bert_pubmed(
			input_ids 	= input_ids,
			attention_mask	= attention_mask
		)

		tok_embed	= out_wiki[0]
		bsz, mtok, dim  = tok_embed.shape
		tok_embed_flat	= tok_embed.reshape(-1, dim)
		men_idx 	= torch.arange(bsz).to(tok_embed.device) * mtok + mention_pos_idx
		wiki_embed 	= tok_embed_flat[men_idx]

		tok_embed	= out_pubmed[0]
		bsz, mtok, dim  = tok_embed.shape
		tok_embed_flat	= tok_embed.reshape(-1, dim)
		men_idx 	= torch.arange(bsz).to(tok_embed.device) * mtok + mention_pos_idx
		pubmed_embed 	= tok_embed_flat[men_idx]


		if   self.p.comb_opn == 'concat': 	pooled_output = torch.cat([wiki_embed, pubmed_embed], dim=1)
		elif self.p.comb_opn == 'add':		pooled_output = wiki_embed + pubmed_embed
		else: raise NotImplementedError

		pooled_output	= self.dropout(pooled_output)
		logits		= self.classifier(pooled_output)
		loss 		= F.binary_cross_entropy_with_logits(logits, labels.float())

		return loss, logits