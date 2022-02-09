import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

class MyModel(nn.Module):
    def __init__(self, model_path='roberta-base', freeze_bert=False):
        super(MyModel, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_path, output_hidden_states=True,
                                               output_attentions=True, return_dict=True)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        # share layer
        # self.softmax_all_layer = nn.Softmax(-1)
        # self.nn_dense = nn.Linear(self.model.config.hidden_size, 1)
        # # use a truncated_normalizer to initialize the α.
        # self.truncated_normal_(self.nn_dense.weight)
        # self.act = nn.ReLU()
        # self.pooler = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        # self.pooler_activation = nn.Tanh()

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.config.hidden_size, 2)
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.model.config.hidden_size, 7),
            nn.Sigmoid()
        )

    # this function is adapted form https://zhuanlan.zhihu.com/p/83609874
    def truncated_normal_(self, tensor, mean=0, std=0.02):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        ## calculate α_i
        # layer_logits = []
        # for layer in outputs.hidden_states[1:]:
        #     out = self.nn_dense(layer)
        #     layer_logits.append(self.act(out))

        # sum up layers by weighting
        # layer_logits = torch.cat(layer_logits, axis=2)
        # layer_dist = self.softmax_all_layer(layer_logits)
        # seq_out = torch.cat([torch.unsqueeze(x, axis=2) for x in outputs.hidden_states[1:]], axis=2)
        # all_layer_output = torch.matmul(torch.unsqueeze(layer_dist, axis=2), seq_out)
        # all_layer_output = torch.squeeze(all_layer_output, axis=2)
        # # take the [CLS] token output
        # all_layer_output = self.pooler_activation(
        #     self.pooler(all_layer_output[:, 0])) if self.pooler is not None else None
        #
        # pooled_output = all_layer_output
        pooled_output = outputs.pooler_output

        output = self.fc1(pooled_output)
        output1 = self.fc2(pooled_output)
        return output, output1