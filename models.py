import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class Bert_pooler(nn.Module):
    def __init__(self, model_path='roberta-base', freeze_bert=False, drop_num=4, drop_prob=0.2, reduction='sum'):
        super(Bert_pooler, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_path, output_hidden_states=True,
                                               output_attentions=True, return_dict=True)

        self.bert_pooler = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Tanh()
        )

        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        self.Loss_Wrapper = Drop_Wrapper(self.model.config.hidden_size, 2, drop_num, drop_prob, reduction)

    def forward(self, input_ids, attention_mask, fwd_type=0, embed=None, out_mtl_drop=False):
        if fwd_type == 2:
            assert embed is not None
            outputs = self.model(input_ids=None, attention_mask=attention_mask, inputs_embeds=embed)
        elif fwd_type == 1:
            token_type_ids = torch.zeros_like(input_ids)
            return self.model.embeddings(input_ids, token_type_ids)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        last_hidden_layer = outputs.last_hidden_state
        output, begin_logic, end_logic = self.Loss_Wrapper(self.bert_pooler(last_hidden_layer[:,0]))
        if out_mtl_drop:
            return output, begin_logic, end_logic
        else:
            return output


class Bert_Lstm_Gru(nn.Module):
    def __init__(self, model_path='roberta-base', freeze_bert=False, rnn_hidden_size=512, drop_num=4, drop_prob=0.2):
        super(Bert_Lstm_Gru, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_path, output_hidden_states=True,
                                               output_attentions=True, return_dict=True)

        self.rnn_hidden_size = rnn_hidden_size
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        self.pooler = nn.Sequential(
            nn.Linear(self.model.config.hidden_size + 2*self.rnn_hidden_size, self.model.config.hidden_size),
            nn.LeakyReLU()
        )

        self.bert_pooler = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(self.model.config.hidden_size, rnn_hidden_size, num_layers=1, bidirectional=True,
                            batch_first=True).cuda()
        self.gru = nn.GRU(rnn_hidden_size*2, 512, num_layers=1, bidirectional=True, batch_first=True).cuda()

        self.drop_wrapper = Drop_Wrapper(self.model.config.hidden_size, 2, drop_num, drop_prob)

    def forward(self, input_ids, attention_mask, fwd_type=0, embed=None, out_mtl_drop=False):
        if fwd_type == 2:
            assert embed is not None
            outputs = self.model(input_ids=None, attention_mask=attention_mask, inputs_embeds=embed)
        elif fwd_type == 1:
            token_type_ids = torch.zeros_like(input_ids)
            return self.model.embeddings(input_ids, token_type_ids)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        last_hidden_layer = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_layer)
        gru_out, gru_h = self.gru(lstm_out)
        gru_h = gru_h.view(-1, 2 * self.rnn_hidden_size)

        pooled_out = self.bert_pooler(last_hidden_layer[:, 0])
        hidden_output = torch.cat([pooled_out, gru_h], dim=1)
        pooled_output = self.pooler(hidden_output)

        output, begin_logic, end_logic = self.drop_wrapper(pooled_output)
        if out_mtl_drop:
            return output, begin_logic, end_logic
        else:
            return output


class Bert_lastClsSep(nn.Module):
    def __init__(self, model_path='roberta-base', freeze_bert=False, with_pooler=True, drop_num=4, drop_prob=0.2):
        super(Bert_lastClsSep, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_path, output_hidden_states=True,
                                               output_attentions=True, return_dict=True)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        self.with_pooler = with_pooler
        if with_pooler:
            self.pooler = nn.Sequential(
                nn.Linear(self.model.config.hidden_size * 3, self.model.config.hidden_size),
                nn.LeakyReLU()
            )
        else:
            self.pooler = nn.Sequential(
                nn.Linear(self.model.config.hidden_size * 2, self.model.config.hidden_size),
                nn.LeakyReLU()
            )

        self.drop_wrapper = Drop_Wrapper(self.model.config.hidden_size, drop_num=drop_num, drop_prob=drop_prob)

    def forward(self, input_ids, attention_mask, fwd_type=0, embed=None, out_mtl_drop=False):
        if fwd_type == 2:
            assert embed is not None
            outputs = self.model(input_ids=None, attention_mask=attention_mask, inputs_embeds=embed)
        elif fwd_type == 1:
            token_type_ids = torch.zeros_like(input_ids)
            return self.model.embeddings(input_ids, token_type_ids)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        last_hidden_layer = outputs.last_hidden_state
        if self.with_pooler:
            hidden_output = torch.cat([outputs.pooler_output, last_hidden_layer[:,0],last_hidden_layer[:,-1]], dim=1)
        else:
            hidden_output = torch.cat([last_hidden_layer[:, 0], last_hidden_layer[:, -1]], dim=1)
        pooled_output = self.pooler(hidden_output)

        output, begin_logic, end_logic = self.drop_wrapper(pooled_output)
        if out_mtl_drop:
            return output, begin_logic, end_logic
        else:
            return output


class Bert_last2cls(nn.Module):
    def __init__(self, model_path='roberta-base', freeze_bert=False, drop_num=4, drop_prob=0.2):
        super(Bert_last2cls, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_path, output_hidden_states=True,
                                               output_attentions=True, return_dict=True)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        self.pooler = nn.Sequential(
            nn.Linear(self.model.config.hidden_size * 2, self.model.config.hidden_size),
            nn.LeakyReLU()
        )

        self.drop_wrapper = Drop_Wrapper(self.model.config.hidden_size, n_class=2, drop_num=drop_num, drop_prob=drop_prob)

    def forward(self, input_ids, attention_mask, fwd_type=0, embed=None, out_mtl_drop=False):
        if fwd_type == 2:
            assert embed is not None
            outputs = self.model(input_ids=None, attention_mask=attention_mask, inputs_embeds=embed)
        elif fwd_type == 1:
            token_type_ids = torch.zeros_like(input_ids)
            return self.model.embeddings(input_ids, token_type_ids)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        hidden_output = torch.cat([outputs.hidden_states[-1][:, 0],
                                   outputs.hidden_states[-2][:, 0]], dim=1)
        pooled_output = self.pooler(hidden_output)

        output, begin_logic, end_logic = self.drop_wrapper(pooled_output)
        if out_mtl_drop:
            return output, begin_logic, end_logic
        else:
            return output


class Bert_all_layer(nn.Module):
    def __init__(self, model_path='roberta-base', freeze_bert=False, drop_num=4, drop_prob=0.2):
        super(Bert_all_layer, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_path, output_hidden_states=True,
                                               output_attentions=True, return_dict=True)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        self.softmax_all_layer = nn.Softmax(-1)
        self.learnable_weight = nn.Linear(self.model.config.hidden_size, 1)
        self.truncated_normal_(self.learnable_weight.weight)

        self.pooler = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Tanh()
        )

        self.fc = nn.Linear(self.model.config.hidden_size, 2)

        self.drop_wrapper = Drop_Wrapper(self.model.config.hidden_size, 2, drop_num, drop_prob)

    def truncated_normal_(self, tensor, mean=0, std=0.02):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def forward(self, input_ids, attention_mask, fwd_type=0, embed=None, out_mtl_drop=False):
        if fwd_type == 2:
            assert embed is not None
            outputs = self.model(input_ids=None, attention_mask=attention_mask, inputs_embeds=embed)
        elif fwd_type == 1:
            token_type_ids = torch.zeros_like(input_ids)
            return self.model.embeddings(input_ids, token_type_ids)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        layer_weight = []
        for layer in outputs.hidden_states[1:]:
            out = self.learnable_weight(layer[:, 0])
            layer_weight.append(F.leaky_relu(out))
        # Hid_num * [batch, 1]
        layer_weight = torch.cat(layer_weight, axis=1)
        # [batch, Hid_num]
        layer_weight = self.softmax_all_layer(layer_weight)
        # [batch, Hid_num]
        seq_out = torch.cat([torch.unsqueeze(x[:, 0], axis=1) for x in outputs.hidden_states[1:]], axis=1)
        # [batch, Hid_num, hidd_size]
        all_layer_output = torch.matmul(torch.unsqueeze(layer_weight, axis=1), seq_out)
        # [batch, 1, Hid_num] matmul [batch, Hid_num, hidd_size]
        # = [batch, 1, hidd_size]
        all_layer_output = torch.squeeze(all_layer_output, axis=1)
        pooled_output = self.pooler(all_layer_output)

        output, begin_logic, end_logic = self.drop_wrapper(pooled_output)
        if out_mtl_drop:
            return output, begin_logic, end_logic
        else:
            return output


class Drop_Wrapper(nn.Module):
    def __init__(self, fc_input_size, n_class=2, drop_num=4, drop_prob=0.2, reduction='sum'):
        super(Drop_Wrapper, self).__init__()
        self.drop_num = drop_num
        self.fc = nn.Linear(fc_input_size, n_class)
        self.drop_out = nn.Dropout(drop_prob)
        self.reduction = reduction

    def forward(self, x):
        loss_stack = torch.stack([
            self.fc(self.drop_out(x)) for _ in range(self.drop_num)
        ], dim=0)
        begin_logic, end_logic = loss_stack[0], loss_stack[-1]
        if self.reduction == 'sum':
            output = torch.sum(loss_stack, dim=0)
        else:
            output = torch.mean(loss_stack, dim=0)
        return output, begin_logic, end_logic
