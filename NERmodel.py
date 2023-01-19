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
                 dropout=0.1, device='cuda', use_lstm=True):
        super(NERmodelbase, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(self.encoder_model)
        # self.ratio = torch.nn.init.uniform_(torch.nn.Parameter(torch.randn([2])))
        if use_lstm:
            self.rnn_dim = self.encoder.config.hidden_size // 2
            self.birnn = nn.LSTM(256, self.rnn_dim, num_layers=1, bidirectional=True,
                                 batch_first=True)
            self.w_omega = nn.Parameter(torch.Tensor(self.encoder.config.hidden_size + (self.rnn_dim * 2), 1))  # 用于门机制学习的矩阵
            nn.init.uniform(self.w_omega, -0.1, 0.1)

        self.ff1 = nn.Linear(in_features=self.encoder.config.hidden_size,
                             out_features=self.encoder.config.hidden_size//2)
        self.ff = nn.Linear(in_features=self.encoder.config.hidden_size//2,
                            out_features=len(self.id_to_tag))
        self.crf = ConditionalRandomField(num_tags=len(self.id_to_tag),
                                          constraints=allowed_transitions('BIO',
                                                                          labels=self.id_to_tag))

        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.spanf1 = SpanF1()
        self.device = device
        self.fl = FocalLoss(alpha=2, gamma=5, reduction='mean')
        self.use_lstm = use_lstm

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
        if self.use_lstm:
            bilstm_out, _ = self.birnn(lstm_encoded)
            temp = torch.cat([embedded_text_input, bilstm_out], dim=-1)
            ratio = torch.sigmoid(torch.matmul(temp, self.w_omega))
            embedded_text_input = ratio * embedded_text_input + (1 - ratio) * bilstm_out
        # print(embedded_text_input.shape, bilstm_out.shape, ratio.shape)
        # project the token representation for classification
        token_score = self.ff1(embedded_text_input)
        token_scores = self.ff(token_score)
        focal_loss = self.fl(token_scores.permute(0, 2, 1), tags)
        # print(focal_loss)
        # print(tags.shape, token_scores.shape)
        token_scores = F.log_softmax(token_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        # print(tags)
        output = self._compute_token_tags(token_scores=token_scores, mask=mask, tags=tags,
                                          metadata=metadata, batch_size=base_shape, mode=mode)
        # print(self.w_omega)

        # temp_val = torch.cat([output['loss'].clone().unsqueeze(0).detach(), torch.tensor(focal_loss).unsqueeze(0).to('cuda')], dim=-1)
        # collective_loss = temp_val * self.ratio

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
        # print("Inside model", self.spanf1.get_metric())
        output = {"loss": loss, "results": self.spanf1.get_metric()}

        if mode == 'predict':
            output['token_tags'] = pred_tags
        return output

#
# if __name__ == '__main__':
#     model = NERmodelbase(tag_to_id=mconern, device=device, encoder_model=encoder_model, dropout=0.3).to(device)
