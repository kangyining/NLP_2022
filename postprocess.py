import os
from collections import Counter

from scipy import stats
import numpy as np
import pandas as pd

def labels2file(p, outf_path, task2=False):
    if not task2:
        with open(outf_path,'w') as outf:
            for pi in p:
                outf.write(','.join([str(k) for k in pi])+'\n')
    else:
        with open(outf_path,'w') as outf:
            for pi in p:
                out = ','.join(str(k) for k in pi)
                outf.write(out+'\n')

def label_filter(a,b):
    if a > b or b < 0.85:
        return 0
    else:
        return 1

# -----------change the predict Threshold ----------------------
# logits = pd.read_csv('./logits.txt', header=None)
# text = pd.read_csv('./data/final_test_set.csv')
# text = text.text.to_numpy()
# filter = (logits==1).to_numpy().flatten()
# text = text[filter]
# pseudo_labeled = []
# par_id = []
# for i, _ in enumerate(text):
#     par_id.append(20000+i)
#     pseudo_labeled.append([1,0,0,0,0,0,0,0])
#
# pseudo_text = pd.DataFrame({'par_id':par_id, 'text':text, 'label':pseudo_labeled})
# pseudo_text.to_csv('pseudo_text')
# new_text = text[]
# logits = logits.apply(lambda x: label_filter(x[0],x[1]), axis=1)
# print(logits.value_counts())

# labels2file([[k] for k in logits.tolist()], 'logits.txt')
# -------------------------------end---------------------------------------

# ----------------------voting--------------------------------
esamble_dir = './ensamble/f1/'
files = os.listdir((esamble_dir))
files.sort(reverse=True)
print(files)
print(len(files))
label_list = []

for i in range(0,4):
    tmp = []
    with open(esamble_dir + files[i]) as f:
        for line in f.readlines():
            label, _ = line.split('\n')
            tmp.append(label)
    label_list.append(tmp)
label_list = np.array(label_list)
mode_list = stats.mode(label_list,axis=0)[0].flatten()

# major vote (The highest F1 will be the final judge)
# mode_list = []
# num = len(label_list)
# for i in range(len(label_list[0])):
#     ls = [int(x) for x in label_list[:, i]]
#     s = sum(ls)
#     if num%2==0:
#         if s > num//2:
#             mode_list.append(1)
#         elif s < num//2:
#             mode_list.append(0)
#         else:
#             mode_list.append(ls[0])
#     else:
#         if s>num//2:
#             mode_list.append(1)
#         else:
#             mode_list.append(0)
print(mode_list)
labels2file([[k] for k in mode_list], 'task1.txt')
