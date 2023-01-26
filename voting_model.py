import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from utils.metric import SpanF1
from utils.reader import extract_spans
from kornia.losses import FocalLoss
from torch.nn import KLDivLoss


class DecoderHead(nn.Module):
    def __init__(self, hidden_size, id_to_tag):
        super(DecoderHead, self).__init__()
        self.id_to_tag = id_to_tag
        # self.rnn_dim = hidden_size // 2
        # self.birnn = nn.LSTM(14, self.rnn_dim, num_layers=1, bidirectional=True,
        #                      batch_first=True)
        # # self.birnn = nn.Linear(14, self.rnn_dim * 2)
        # self.w_omega = nn.Parameter(
        #     torch.Tensor(hidden_size + (self.rnn_dim * 2), 1))  # 用于门机制学习的矩阵
        # nn.init.uniform_(self.w_omega, -0.1, 0.1)
        self.rnn_dim = hidden_size // 2
        self.birnn = nn.LSTM(hidden_size, self.rnn_dim, num_layers=1, bidirectional=True,
                             batch_first=True)
        # self.birnn = nn.Linear(256, self.rnn_dim * 2)
        self.w_omega = nn.Parameter(
            torch.Tensor(hidden_size + (self.rnn_dim * 2), 1))  # 用于门机制学习的矩阵
        nn.init.uniform_(self.w_omega, -0.1, 0.1)

        self.ff1 = nn.Linear(in_features=hidden_size,
                             out_features=hidden_size // 2)
        self.ff = nn.Linear(in_features=hidden_size // 2,
                            out_features=len(self.id_to_tag))
        self.feature_enc = nn.Linear(72, hidden_size)
        self.fl = FocalLoss(alpha=2, gamma=5, reduction='mean')

    def forward(self, embedded_text_input, lstm_encoded, tags):
        tags_encoded = self.feature_enc(lstm_encoded)
        bilstm_out, _ = self.birnn(tags_encoded)
        temp = torch.cat([embedded_text_input, bilstm_out], dim=-1)
        ratio = torch.sigmoid(torch.matmul(temp, self.w_omega))
        embedded_text_input = ratio * embedded_text_input + (1 - ratio) * bilstm_out
        token_score = self.ff1(embedded_text_input)
        token_scores = self.ff(token_score)
        focal_loss = self.fl(token_scores.permute(0, 2, 1), tags)

        # token_scores = F.log_softmax(token_scores, dim=-1)
        return focal_loss, token_scores


class NERmodelbase3(nn.Module):
    def __init__(self, encoder_model='xlm-roberta-large', tag_to_id=None, lr=1e-3,
                 dropout=0.1, device='cuda', use_lstm=True):
        super(NERmodelbase3, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(self.encoder_model)
        # self.encoder2 = AutoModel.from_pretrained(self.encoder_model, hidden_dropout_prob=0.3,
        #                                           attention_probs_dropout_prob=0.25)
        self.crf = ConditionalRandomField(num_tags=len(self.id_to_tag),
                                          constraints=allowed_transitions('BIO',
                                                                          labels=self.id_to_tag))
        self.head_1 = DecoderHead(self.encoder.config.hidden_size, self.id_to_tag)
        self.head_2 = DecoderHead(self.encoder.config.hidden_size, self.id_to_tag)
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.spanf1 = SpanF1()
        self.device = device
        # self.fl = FocalLoss(alpha=2, gamma=5, reduction='mean')
        self.use_lstm = use_lstm
        self.kl = KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, x, mode=''):
        tokens, tags, mask, token_mask, metadata, lstm_encoded = x
        # print(tokens, mask, token_mask, metadata, tags)
        tokens, mask, token_mask, tags, lstm_encoded = tokens.to(self.device), mask.to(self.device), \
                                                       token_mask.to(self.device), tags.to(self.device), \
                                                       lstm_encoded.to(self.device)

        base_shape = tokens.size(0)
        embedded_text = self.encoder(input_ids=tokens, attention_mask=mask, output_hidden_states=True)
        embedded_text_input = embedded_text.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # embedded_text_input_2 = self.encoder2(input_ids=tokens, attention_mask=mask)
        embedded_text_input_2 = torch.stack(embedded_text.hidden_states[-8:], dim=-1).mean(-1)
        embedded_text_input_2 = self.dropout(F.leaky_relu(embedded_text_input_2))

        fl1, token_score_1 = self.head_1(embedded_text_input, lstm_encoded, tags)
        fl2, token_score_2 = self.head_2(embedded_text_input_2, lstm_encoded, tags)
        # compute the log-likelihood loss and compute the best NER annotation sequence
        # print(tags)
        # token_scores = torch.stack([token_score_1, token_score_2]).mean(dim=0)
        token_scores = (token_score_2 + token_score_1) / 2
        # Soft voting scores
        # target = token_scores.clone()
        token_scores = F.log_softmax(token_scores, dim=-1)

        output, all_prob = self._compute_token_tags(token_scores=token_scores, mask=mask, tags=tags,
                                                    metadata=metadata, batch_size=base_shape, mode=mode)
        focal_loss = (fl1 + fl2) / 2
        # Calculate the KL divergence score for both the models
        kl1 = self.kl(F.log_softmax(token_score_1, dim=-1), token_scores)
        kl2 = self.kl(F.log_softmax(token_score_2, dim=-1), token_scores)
        kl_score = (kl1 + kl2) / 2
        return output, focal_loss, kl_score

    def _compute_token_tags(self, token_scores, mask, tags, metadata, batch_size, mode=''):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf(token_scores, tags, mask) / float(batch_size)
        best_path, all_prob = self.crf.viterbi_tags(token_scores, mask)

        pred_results, pred_tags = [], []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            # print(tag_seq)
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))
        # print(pred_tags, pred_results)
        # print("Val", pred_results, metadata)
        self.spanf1(pred_results, metadata)
        # print("Inside model", self.spanf1.get_metric())
        output = {"loss": loss, "results": self.spanf1.get_metric()}

        if mode == 'predict':
            output['token_tags'] = pred_tags
        return output, all_prob

#
# if __name__ == '__main__':
#     model = NERmodelbase(tag_to_id=mconern, device=device, encoder_model=encoder_model, dropout=0.3).to(device)
