import pandas as pd
from ast import literal_eval
import os

# read the whole data set -- adapted from https://github.com/Perez-AlmendrosC/dontpatronizeme
def load_task1():
    """
    Load task 1 training set and convert the tags into binary labels.
    Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
    Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
    It returns a pandas dataframe with paragraphs and labels.
    """
    rows=[]
    with open(r'../data/dontpatronizeme_pcl.tsv', "r") as f:
        for line in f.readlines()[4:]:
            par_id=line.strip().split('\t')[0]
            art_id = line.strip().split('\t')[1]
            keyword=line.strip().split('\t')[2]
            country=line.strip().split('\t')[3]
            t=line.strip().split('\t')[4]#.lower()
            l=line.strip().split('\t')[-1]
            if l=='0' or l=='1':
                lbin=0
            else:
                lbin=1
            rows.append(
                {'par_id':par_id,
                'art_id':art_id,
                'keyword':keyword,
                'country':country,
                'text':t,
                'label':lbin,
                'orig_label':l
                }
                )
    df=pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label'])
    return df

def load_test():
    #self.test_df = [line.strip() for line in open(self.test_path)]
    rows=[]
    with open(r'../data/task4_test.tsv', "r") as f:
        for line in f:
            t=line.strip().split('\t')
            rows.append(t)
    test_set_df = pd.DataFrame(rows, columns="par_id art_id keyword country text".split())
    return test_set_df

# whole dataset
dataset = load_task1()
trids = pd.read_csv('../data/train_semeval_parids-labels.csv')
teids = pd.read_csv('../data/dev_semeval_parids-labels.csv')
trids.par_id = trids.par_id.astype(str)
teids.par_id = teids.par_id.astype(str)
# Rebuild training set (Task 1)
rows = []  # will contain par_id, label and text
for idx in range(len(trids)):
    parid = trids.par_id[idx]
    label1 = trids.label[idx]
    # print(parid)
    # select row from original dataset to retrieve `text` and binary label
    text = dataset.loc[dataset.par_id == parid].text.values[0]
    label = dataset.loc[dataset.par_id == parid].label.values[0]
    mixed_label = '[' + str(label) + ', ' + label1[1:]
    rows.append({
        'par_id': parid,
        'text': text,
        'label': mixed_label
    })
trdf1 = pd.DataFrame(rows)
print(trdf1.head())
# Rebuild test set (Task 1)
rows = []  # will contain par_id, label and text
for idx in range(len(teids)):
    parid = teids.par_id[idx]
    label1 = teids.label[idx]
    # print(parid)
    # select row from original dataset
    text = dataset.loc[dataset.par_id == parid].text.values[0]
    label = dataset.loc[dataset.par_id == parid].label.values[0]
    mixed_label = '[' + str(label) + ', ' + label1[1:]
    rows.append({
        'par_id': parid,
        'text': text,
        'label': mixed_label
    })

tedf1 = pd.DataFrame(rows)

final_test = load_test()
final_test = final_test[['par_id', 'text']]


print(len(trdf1))
print(len(tedf1))
print(len(final_test))
trdf1.to_csv("../data/training_set.csv")
tedf1.to_csv("../data/test_set.csv")
final_test.to_csv("../data/final_test_set.csv")