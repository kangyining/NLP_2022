import random
from ast import literal_eval
from collections import Counter
import numpy as np
import torch

# from googletrans import Translator
# import nlpaug.augmenter.char as nac
# import nlpaug.augmenter.word as naw
# import nlpaug.augmenter.sentence as nas
# from nlpaug.util import Action
# import nltk
# from nltk.corpus import stopwords
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

# -----------load train and val data------------------------------

# train_data = pd.read_csv("./data/training_set.csv")
# train_data.label = train_data.label.apply(literal_eval)
#
# test_data = pd.read_csv("./data/test_set.csv")
# test_data.label = test_data.label.apply(literal_eval)
# final_data = pd.read_csv('./data/final_test_set.csv')
# train_data = train_data[['par_id','text','label']]
# all_pos = train_data[train_data.label.apply(lambda x: sum(x) > 0)]

# -----------------------------end--------------------------------

# -----------load back_translation datas--------------------------

# # remove duplicate
# fr = pd.read_csv('./franch.csv', header=None, index_col=0, names=['text'])
# fr = pd.DataFrame({'par_id': all_pos.par_id, 'text': fr.text, 'label': all_pos.label})
# concat = pd.concat([all_pos, fr], axis=0, ignore_index=True)
# fr = concat.drop_duplicates(['text'], keep='first', )[len(all_pos):]
# sp = pd.read_csv('./spanish.csv', header=None, index_col=0, names=['text'])
# sp = pd.DataFrame({'par_id': all_pos.par_id, 'text': sp.text, 'label': all_pos.label})
# concat = pd.concat([all_pos,sp])
# sp = concat.drop_duplicates(['text'], keep='first')[len(all_pos):]
# de_fr = pd.read_csv('./data/de_fr_translate', index_col=0)
# de_fr = pd.DataFrame({'par_id': de_fr.par_id, 'text': de_fr.text, 'label': de_fr.label})
# concat = pd.concat([all_pos,de_fr])
# de_fr = concat.drop_duplicates(['text'], keep='first')[len(all_pos):]
#
# # remove duplicate
# mix_tran = pd.concat([fr,sp])
# mix_tran = mix_tran.drop_duplicates(['text'], keep='first')
# fr = mix_tran[:len(fr)]
# sp = mix_tran[len(fr):]
# mix_tran = pd.concat([fr, sp, de_fr])
# mix_tran = mix_tran.drop_duplicates(['text'], keep='first')
# de_fr = mix_tran[(len(fr)+len(sp)):]
#
# fr.to_csv('fr_translate', index=False)
# sp.to_csv('sp_translate', index=False)
# de_fr.to_csv('de_fr_translate', index=False)

# ----------------------------------end--------------------------------------


# ------------------data argumentation----------------------------------------

# print(len(all_pos.text.tolist()))
# all_pos = all_pos.text.tolist()
# print(len(all_pos))
# pd.DataFrame(all_pos).to_csv('pos_text')
## BackTranslationAug with facebook model
# back_translation_aug = naw.BackTranslationAug(
#     from_model_name='facebook/wmt19-en-de',
#     to_model_name='facebook/wmt19-de-en',
#     max_length=600,
#     device=device,
#     batch_size=4
# )
#
## Synonym word replacement
# words = stopwords.words('english')
# aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2, aug_max=100, stopwords=words)
# #
# text = []
# label = []
# text_par_id = []
# Synonym = []
# Synonym_label = []
# Synonym_par_id = []
# for _, row in train_data.iterrows():
#     Synonym.append(row['text'])
#     Synonym_label.append(row['label'])
#     Synonym_par_id.append(row['par_id'])
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
# Syn_data.to_csv('Synonym_data', index=None)
# aug_data.to_csv('back_translation_data')

# ------------------------------------------end--------------------------------------------

# ----------test model output---------

# train_data_loader = create_data_loader(train_data, True, 'roberta-base', 512, 4,shuffle=True)
# models = MyModel()
# batch = next(iter(train_data_loader))
# input_ids = batch["input_ids"]
# attention_mask = batch["attention_mask"]
# output = models(input_ids, attention_mask)
# print(output)

# ----------------end-----------------

# ----------------show the frequency of tokens length-----------------
# token_counts = []
# tokenizer = AutoTokenizer.from_pretrained('roberta-base', output_hidden_states=True, return_dict=True)
# for _, row in train_data.iterrows():
#     count = tokenizer.encode(
#         row['text'],
#         max_length=512,
#         truncation='do_not_truncate'
#     )
#     token_counts.append(len(count))
# print(sorted(token_counts,reverse=True))
# sns.histplot(token_counts)
# plt.show()

# --------------------------end--------------------------------------


# -------------------------Filtering easy argm data ---------------------------
# def label_filter(a,b):
#     if  a>b:
#         return 0
#     else:
#         return 1

# syno_train_data = pd.read_csv('./data/Synonym_data')
# syno_train_data.label = syno_train_data.label.apply(literal_eval)
# fr_trans_train_data = pd.read_csv('./data/fr_translate')
# fr_trans_train_data.label = fr_trans_train_data.label.apply(literal_eval)
# sp_trans_train_data = pd.read_csv('./data/sp_translate')
# sp_trans_train_data.label = sp_trans_train_data.label.apply(literal_eval)
# de_fr_trans_train_data = pd.read_csv('./data/de_fr_translate')
# de_fr_trans_train_data.label = de_fr_trans_train_data.label.apply(literal_eval)
# train_data = pd.concat([sp_trans_train_data,syno_train_data, fr_trans_train_data, de_fr_trans_train_data])
# train_data = pd.DataFrame({'par_id': train_data.par_id, 'text': train_data.text, 'label': train_data.label})
# test_data = pd.read_csv('./data/test_set.csv')
# test_data.label = test_data.label.apply(literal_eval)
# logits = pd.read_csv('./logits1.txt', header=None, index_col=None)
# print(len(logits))
# logits = logits.apply(lambda x: label_filter(x[0],x[1]), axis=1)
# print(logits.value_counts())
# # print(logits)
#
# all_pos = test_data[test_data.label.apply(lambda x: sum(x) > 0) & (logits==0)]
# all_pos = all_pos.text.tolist()
# print(len(all_pos))
# print(sorted([len(tex) for tex in all_pos]))
# print(all_pos[-1])


# print(train_data[logits==0])
# train_data[logits==0].to_csv('hard', index=None)

# --------------------------------end-------------------------------
