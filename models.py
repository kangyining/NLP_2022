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

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.config.hidden_size, 2)
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.model.config.hidden_size, 7),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, fwd_type=0, embed=None):
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

        pooled_output = outputs.pooler_output

        output = self.fc1(pooled_output)
        output1 = self.fc2(pooled_output)
        # return output
        return output, output1