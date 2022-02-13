import random
from ast import literal_eval
from collections import Counter
import numpy as np
import torch

# import nlpaug.augmenter.char as nac
# import nlpaug.augmenter.word as naw
# import nlpaug.augmenter.sentence as nas
# from nlpaug.util import Action
# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
# import seaborn as sns
from data_processing import create_data_loader

torch.cuda.current_device()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from models import MyModel

train_data = pd.read_csv("./data/training_set.csv")
train_data.label = train_data.label.apply(literal_eval)
# test_data = pd.read_csv("./data/test_set.csv")
# final_data = pd.read_csv('./data/final_test_set.csv')
# train_data = train_data[['par_id','text','label']]
# train_data.to_csv('arg_train_data')

# all_pos = train_data[train_data.label.apply(lambda x: sum(x) > 0)]
# back_translation_aug = naw.BackTranslationAug(
#     from_model_name='facebook/wmt19-en-de',
#     to_model_name='facebook/wmt19-de-en',
#     max_length=600,
#     device=device,
#     batch_size=4
# )
# aug = naw.SynonymAug(aug_src='wordnet', aug_max=50, aug_p=0.2)
# #
# text = []
# label = []
# text_par_id = []
# Synonym = []
# Synonym_label = []
# Synonym_par_id = []
# for _, row in all_pos.iterrows():
#     if random.random() >0.8:
#         Synonym.append(row['text'])
#         Synonym_label.append(row['label'])
#         Synonym_par_id.append(row['par_id'])
#     text.append(row['text'])
#     label.append(row['label'])
#     text_par_id.append(row['par_id'])
# #
# text = back_translation_aug.augment(text, num_thread=1)
# aug_text = aug.augment(Synonym)
# # print(text)
# aug_data = pd.DataFrame({'par_id': text_par_id, 'text': text, 'label': label})
# Syn_data = pd.DataFrame({'par_id': Synonym_par_id, 'text': aug_text, 'label': Synonym_label})
# # print(aug_data)
# #
# Syn_data.to_csv('Synonym_data')
# aug_data.to_csv('back_translation_data')

# count = [0,0,0,0,0,0,0]
# for label in all_pos.label:
#     print(label)
#     for i,value in enumerate(label):
#         if i != 0:
#             count[i-1] += int(value)
# print(count)
# count = [1/i for i in count]
# su = sum(count)
# count = [i/su for i in count]
# print(count)

# train_data_loader = create_data_loader(train_data, True, 'roberta-base', 512, 4,shuffle=True)
models = MyModel()
# batch = next(iter(train_data_loader))
# input_ids = batch["input_ids"]
# attention_mask = batch["attention_mask"]
# output = models(input_ids, attention_mask, fwd_type=1)
# print(output)
for name, param in models.named_parameters():
    if param.requires_grad:
        print(name)

# token_counts = []
# tokenizer = AutoTokenizer.from_pretrained('roberta-base', output_hidden_states=True, return_dict=True)
# for _, row in test_data.iterrows():
#     count = tokenizer.encode(
#         row['text'],
#         max_length=512,
#         truncation='do_not_truncate'
#     )
#     token_counts.append(len(count))
# print(sorted(token_counts,reverse=True))
# sns.histplot(token_counts)
# plt.show()