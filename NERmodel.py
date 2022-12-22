import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from utils.metric import SpanF1
from utils.reader import extract_spans


class NERmodelbase(nn.Module):
    def __init__(self, encoder_model='xlm-roberta-large', tag_to_id=None, lr=1e-3,
                 dropout=0.1, device='cuda'):
        super(NERmodelbase, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(self.encoder_model)
        self.ff = nn.Linear(in_features=self.encoder.config.hidden_size,
                            out_features=len(self.id_to_tag))
        self.crf = ConditionalRandomField(num_tags=len(self.id_to_tag),
                                          constraints=allowed_transitions('BIO',
                                                                          labels=self.id_to_tag))
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.spanf1 = SpanF1()
        self.device = device

    def forward(self, x, mode=''):
        tokens, tags, mask, token_mask, metadata = x
        tokens, mask, token_mask, tags = tokens.to(self.device), mask.to(self.device),\
                                         token_mask.to(self.device), tags.to(self.device)
        base_shape = tokens.size(0)
        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.ff(embedded_text_input)
        token_scores = F.log_softmax(token_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(token_scores=token_scores, mask=mask, tags=tags,
                                          metadata=metadata, batch_size=base_shape, mode=mode)
        return output

    def _compute_token_tags(self, token_scores, mask, tags, metadata, batch_size, mode=''):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf(token_scores, tags, mask) / float(batch_size)
        best_path = self.crf.viterbi_tags(token_scores, mask)

        pred_results, pred_tags = [], []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.spanf1(pred_results, metadata)
        output = {"loss": loss, "results": self.spanf1.get_metric()}

        if mode == 'predict':
            output['token_tags'] = pred_tags
        return output
