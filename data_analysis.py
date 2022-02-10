from ast import literal_eval
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
# import seaborn as sns

from models import MyModel

train_data = pd.read_csv("./data/training_set.csv")
train_data.label = train_data.label.apply(literal_eval)
# test_data = pd.read_csv("./data/test_set.csv")
# final_data = pd.read_csv('./data/final_test_set.csv')
all_pos = train_data[train_data.label.apply(lambda x: sum(x) > 0)]
count = [0,0,0,0,0,0,0]
for label in all_pos.label:
    print(label)
    for i,value in enumerate(label):
        if i != 0:
            count[i-1] += int(value)
print(count)
count = [1/i for i in count]
su = sum(count)
count = [i/su for i in count]
print(count)

# models = MyModel()
# for name, param in models.named_parameters():
#     print(name)

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