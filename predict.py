from collections import Counter
import zipfile

import numpy as np
import torch
import pandas as pd

from models import MyModel
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


MODEL_TYPE = 'roberta-large'

test_data = pd.read_csv("./data/final_test_set.csv")
test_data_loader = create_data_loader(test_data, False, MODEL_TYPE, 512, 8)

model = MyModel(model_path=MODEL_TYPE)
# model.load_state_dict(torch.load('./best_model_state.bin'))
model.load_state_dict(torch.load('./best_model_state.bin'))
model = model.to(device)

pred, pred2 = trainer.pridict(model, test_data_loader, device)
print(pred)
print(len(pred))
print(Counter(pred))
# print(pred2)
pred2 = np.array(pred2)
print(pred2.max())
for i in range(len(pred2)):
    if pred[i] == 0:
        pred2[i] = [0, 0, 0, 0, 0, 0, 0]
pred2 = pred2 > 0.2
pred2 = pred2.astype(int)
labels2file([[k] for k in pred], 'task1.txt')
labels2file(pred2, 'task2.txt')
z = zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('task1.txt')
z.write('task2.txt')
z.close()
