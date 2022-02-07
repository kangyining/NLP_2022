from collections import Counter
import zipfile
import torch
import pandas as pd

from models import MyModel
import trainer
from data_processing import create_data_loader

torch.cuda.current_device()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# adapted from https://github.com/Perez-AlmendrosC/dontpatronizeme
def labels2file(p, outf_path):
    with open(outf_path,'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi])+'\n')

MODEL_TYPE = 'roberta-base'

test_data = pd.read_csv("./data/final_test_set.csv")
test_data_loader = create_data_loader(test_data, False, MODEL_TYPE, 512, 4)

model = MyModel()
model.load_state_dict(torch.load('./best_model_state.bin'))
model = model.to(device)

pred, _ = trainer.pridict(model, test_data_loader, device)
print(pred)
print(len(pred))
print(Counter(pred))
labels2file([[k] for k in pred], 'task1.txt')
z = zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('task1.txt')
z.close()