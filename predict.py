from collections import Counter
import zipfile

import numpy as np
import torch
import pandas as pd
from ast import literal_eval

import loss_function
from models import Bert_Lstm_Gru, Bert_last2cls, Bert_pooler, Bert_lastClsSep, Bert_all_layer
import trainer
from data_processing import create_data_loader

torch.cuda.current_device()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# adapted from https://github.com/Perez-AlmendrosC/dontpatronizeme
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


MODEL_TYPE = 'microsoft/deberta-v3-large'

test_data = pd.read_csv("./data/final_test_set.csv")
val_data = pd.read_csv("./data/val_set.csv")
val_data.label = val_data.label.apply(literal_eval)
# syno_train_data = pd.read_csv('./data/Synonym_data')
# syno_train_data.label = syno_train_data.label.apply(literal_eval)
# fr_trans_train_data = pd.read_csv('./data/fr_translate')
# fr_trans_train_data.label = fr_trans_train_data.label.apply(literal_eval)
# sp_trans_train_data = pd.read_csv('./data/sp_translate')
# sp_trans_train_data.label = sp_trans_train_data.label.apply(literal_eval)
# de_fr_trans_train_data = pd.read_csv('./data/de_fr_translate')
# de_fr_trans_train_data.label = de_fr_trans_train_data.label.apply(literal_eval)
# train_data = pd.concat([sp_trans_train_data,syno_train_data, fr_trans_train_data, de_fr_trans_train_data])
# print(len(train_data))
test_data_loader = create_data_loader(test_data, False, MODEL_TYPE, 512, 32)
val_data_loader = create_data_loader(val_data, True, MODEL_TYPE, 512, 32, sampler=None)

model = Bert_lastClsSep(model_path=MODEL_TYPE, with_pooler=False)
model.load_state_dict(torch.load('./best_model_state.bin'))
# model.load_state_dict(torch.load('./8.bin'))
model = model.to(device)

pred, logits = trainer.pridict(model, test_data_loader, device)
loss_fn = []
loss_fn.append((1, loss_function.DiceLoss(alpha=0.5).to(device)))
trainer.evaluate(model, val_data_loader, device, loss_fn)
print(pred)
print(len(pred))
print(Counter(pred))
# print(pred2)
# pred2 = np.array(pred2)
# print(pred2.max())
# for i in range(len(pred2)):
#     if pred[i] == 0:
#         pred2[i] = [0, 0, 0, 0, 0, 0, 0]
# pred2 = pred2 > 0.2
# pred2 = pred2.astype(int)
labels2file([[k] for k in pred], 'task1.txt')
labels2file([k for k in logits], 'logits1.txt')
# labels2file(pred2, 'task2.txt')
z = zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('task1.txt')
# z.write('task2.txt')
z.close()
