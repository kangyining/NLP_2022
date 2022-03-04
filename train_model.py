import argparse
import ast
from ast import literal_eval
import random
import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
from transformers import AutoTokenizer, AdamW
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import numpy as np
import pandas as pd

from models import Bert_Lstm_Gru, Bert_last2cls, Bert_pooler, Bert_lastClsSep, Bert_all_layer
import trainer
from data_processing import create_data_loader
import loss_function

parser = argparse.ArgumentParser(description='Model HyperParameter')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=12, metavar='E',
                    help='number of epochs to train (default: 12)')
parser.add_argument('--lr', type=float, default=5e-6, metavar='LR',
                    help='learning rate (default: 5e-6)')
parser.add_argument('--seed', type=int, default=15, metavar='S',
                    help='random seed (default: 15)')
parser.add_argument('--model', type=str, default='microsoft/deberta-v3-large',
                    help='the name of pre-trained model using in experimrnt like roberta-large (default: microsoft/deberta-v3-large)')
parser.add_argument('--model_output', type=str, default='cls_spe',
                    help='Choosing model output style: lstm_gru, cls_spe, all_layer, last_2_cls, base')
parser.add_argument('--dice_fact', type=float, default=0.3,
                    help='hyper_param for dice loss')
parser.add_argument('--atk', type=str, default='',
                    help='Apply adversarial machine learning: select from FGM and SMART')
parser.add_argument('--ema', type=ast.literal_eval, default=True,
                    help="Apply Exponential Moving Average to improve robust ")
parser.add_argument('--grad_clamp', type=ast.literal_eval, default=True,
                    help="Apply gradients clamping")
parser.add_argument('--r_drop', type=ast.literal_eval, default=True,
                    help="Apply r_drop regularization")
parser.add_argument('--sample_num', type=float, default=2.5,
                    help='the total sample of weighted sampling: 3 mean 3 * positive number')
args = parser.parse_args()

print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
MAX_LEN = 512

torch.cuda.current_device()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    set_seed(args.seed)
    # load_data
    train_data = pd.read_csv("./data/training_set.csv")
    train_data.label = train_data.label.apply(literal_eval)
    # '''   apply data argument
    # syno_armg = pd.read_csv('./data/Synonym_data_all')
    # syno_armg.label = syno_armg.label.apply(literal_eval)
    # back_transfer = pd.read_csv('./data/sp_translate')
    # back_transfer.label = back_transfer.label.apply(literal_eval)
    # train_data = pd.concat([train_data, syno_armg])
    # '''
    val_data = pd.read_csv("./data/val_set.csv")
    val_data.label = val_data.label.apply(literal_eval)

    # ------------if apply cross validation---------------
    # train_data = pd.concat([train_data, test_data, syno_armg, back_transfer])

    # kfold = StratifiedShuffleSplit(n_splits=5, train_size=0.8, random_state=args.seed)
    # X = train_data.text
    # y = train_data.label.tolist()
    # y = [i[0] for i in y]
    # for index, (train_id, test_id) in enumerate(kfold.split(X, y)):
    #     print(index)
    #     train_d = train_data.iloc[train_id]
    #     test_d = train_data.iloc[test_id]
    #     print([len(train_d),len(test_d)])
    #     print(len(test_d))
    #
    #     test_target = test_d.label.tolist()
    #     test_target = [i[0] for i in test_target]
    #     test_class_sample_count = np.array([len(np.where(test_target == t)[0]) for t in np.unique(test_target)])
    #     print(test_class_sample_count)

    # print the distribution of labels for train set
    target = train_data.label.tolist()
    target = [i[0] for i in target]
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    print(class_sample_count)

    # apply weight sampling
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, int(class_sample_count[1]*args.sample_num), replacement=False)
    train_data_loader = create_data_loader(train_data, True, args.model, MAX_LEN, args.batch_size, sampler=sampler, shuffle=False)

    # train_data_loader = create_data_loader(train_data, True, args.model, MAX_LEN, args.batch_size, shuffle=True)
    val_data_loader = create_data_loader(val_data, True, args.model, MAX_LEN, args.batch_size, sampler=None)

    # print(len(train_data_loader), len(val_data_loader))
    if args.model_output == 'base':
        print('use pooler out')
        model = Bert_pooler(model_path=args.model)
    elif args.model_output == 'cls_spe':
        print('use last cls and sep out')
        model = Bert_lastClsSep(model_path=args.model, with_pooler=False)
    elif args.model_output == 'all_layer':
        print('use all layer out')
        model = Bert_all_layer(model_path=args.model)
    elif args.model_output == 'last_2_cls':
        print('use last 2 cls out')
        model = Bert_last2cls(model_path=args.model)
    elif args.model_output == 'lstm_gru':
        print('use lstm gru out')
        model = Bert_Lstm_Gru(model_path=args.model)
    else:
        print('use lstm gru out')
        model = Bert_Lstm_Gru(model_path=args.model)
    model = model.to(device)
    # model.load_state_dict(torch.load('./best_model_state.bin'))
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False, weight_decay=1e-6)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

    loss_fn = []
    loss_fn.append((1, loss_function.DiceLoss(alpha=args.dice_fact).to(device)))
    # loss_fn.append((1, loss_function.FocalLoss(alpha=0.5, gamma=2.0).to(device)))
    # loss_fn.append((1, nn.CrossEntropyLoss().to(device)))
    trainer.train_epoch(model, optimizer, device, loss_fn, train_data_loader, args,
                        val_dataloader=val_data_loader, epochs=args.epochs, scheduler=scheduler, model_id='')

