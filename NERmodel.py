import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from utils.metric import SpanF1
from utils.reader import extract_spans
from kornia.losses import FocalLoss


class NERmodelbase(nn.Module):
    def __init__(self, encoder_model='xlm-roberta-large', tag_to_id=None, lr=1e-3,
                 dropout=0.1, device='cuda'):
        super(NERmodelbase, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(self.encoder_model)
        self.ff = nn.Linear(in_features=self.encoder.config.hidden_size+256,
                            out_features=len(self.id_to_tag))
        self.crf = ConditionalRandomField(num_tags=len(self.id_to_tag),
                                          constraints=allowed_transitions('BIO',
                                                                          labels=self.id_to_tag))
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.spanf1 = SpanF1()
        self.device = device
        self.fl = FocalLoss(alpha=2, reduction='mean')

    def forward(self, x, mode=''):
        tokens, tags, mask, token_mask, metadata, lstm_encoded = x
        # print(tokens, mask, token_mask, metadata, tags)
        tokens, mask, token_mask, tags, lstm_encoded = tokens.to(self.device), mask.to(self.device), \
                                         token_mask.to(self.device), tags.to(self.device), \
                                         lstm_encoded.to(self.device)

        base_shape = tokens.size(0)
        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))
        embedded_text_input = torch.cat([embedded_text_input, lstm_encoded], dim=-1)
        # print(embedded_text_input.shape, lstm_encoded.shape)
        # project the token representation for classification
        token_scores = self.ff(embedded_text_input)
        focal_loss = self.fl(token_scores.permute(0,2,1), tags)
        # print(focal_loss)
        # print(tags.shape, token_scores.shape)
        token_scores = F.log_softmax(token_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        # print(tags)
        output = self._compute_token_tags(token_scores=token_scores, mask=mask, tags=tags,
                                          metadata=metadata, batch_size=base_shape, mode=mode)

        return output, focal_loss

    def _compute_token_tags(self, token_scores, mask, tags, metadata, batch_size, mode=''):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf(token_scores, tags, mask) / float(batch_size)
        best_path = self.crf.viterbi_tags(token_scores, mask)

        pred_results, pred_tags = [], []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            # print(tag_seq)
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))
        # print(pred_tags, pred_results)
        # print("Val", pred_results, metadata)
        self.spanf1(pred_results, metadata)
        output = {"loss": loss, "results": self.spanf1.get_metric()}

        if mode == 'predict':
            output['token_tags'] = pred_tags
        return output

#
# if __name__ == '__main__':
#     model = NERmodelbase(tag_to_id=mconern, device=device, encoder_model=encoder_model, dropout=0.3).to(device)
