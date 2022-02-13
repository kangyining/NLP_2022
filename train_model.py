import argparse
import ast
from ast import literal_eval
import random

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
from transformers import AdamW, AutoTokenizer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from models import MyModel
import trainer
from data_processing import create_data_loader
import loss_function

parser = argparse.ArgumentParser(description='Model HyperParameter')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=15, metavar='E',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=2e-6, metavar='LR',
                    help='learning rate (default: 2e-6)')
parser.add_argument('--seed', type=int, default=150, metavar='S',
                    help='random seed (default: 70)')
parser.add_argument('--model', type=str, default='roberta-large',
                    help='the name of pre-trained model using in experimrnt like albert-xxlarge-v2 (default: roberta-large)')
args = parser.parse_args()

MAX_LEN = 512

torch.cuda.current_device()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    set_seed(args.seed)

    # load_data
    train_data = pd.read_csv("./data/training_set.csv")
    train_data.label = train_data.label.apply(literal_eval)
    # back_trans_train_data = pd.read_csv('./data/pseudo_text')
    # back_trans_train_data.label = back_trans_train_data.label.apply(literal_eval)
    # train_data = pd.concat([train_data, back_trans_train_data])
    test_data = pd.read_csv("./data/test_set.csv")
    test_data.label = test_data.label.apply(literal_eval)

    target = train_data.label.tolist()
    target = [i[0] for i in target]
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    print(class_sample_count)
    # class_sample_count[0] += 2000
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, int(len(samples_weight)/4), replacement=False)
    print(int(len(samples_weight)/4))
    # all_negs = train_data[train_data.label.apply(lambda x: sum(x) == 0)]
    # all_pos = train_data[train_data.label.apply(lambda x: sum(x) > 0)]
    # training_set2 = pd.concat([all_pos, all_negs[:round(len(all_pos) * 0.5)]])
    # training_set2 = all_pos

    # target = training_set2.label.tolist()
    # target = [i[0] for i in target]
    # class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    # print(class_sample_count)

    train_data_loader = create_data_loader(train_data, True, args.model, MAX_LEN, args.batch_size, sampler=sampler, shuffle=False)
    # train_data_loader = create_data_loader(training_set2, True, args.model, MAX_LEN, args.batch_size, shuffle=True)
    val_data_loader = create_data_loader(test_data, True, args.model, MAX_LEN, args.batch_size, sampler=None)

    print(len(train_data_loader), len(val_data_loader))
    model = MyModel(model_path=args.model)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.000003, correct_bias=False)
    # optimizer = AdamW(model.parameters(), lr=0.000001, correct_bias=False)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    weight = torch.Tensor([0.02743, 0.09841, 0.09719, 0.08201, 0.10859, 0.04337, 0.54297])
    loss_fn = []
    loss_fn.append(loss_function.FocalLoss(alpha=0.75, gamma=2.0).to(device))
    loss_fn.append(loss_function.FocalLoss2d(gamma=2.0, weight=weight).to(device))
    # loss_fn.append(nn.BCEWithLogitsLoss(weight=weight).to(device))
    # loss_fn = nn.CrossEntropyLoss()

    trainer.train_epoch(model, optimizer, device, loss_fn, train_data_loader, val_data_loader, 15, scheduler)
