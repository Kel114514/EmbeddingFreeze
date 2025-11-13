#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 19 Sep, 2019

@author: wangshuo
"""

import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from utils import collate_fn
from narm import NARM
from dataset import load_data, RecSysDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/diginetica/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_path = f"./embeddings/{args.dataset_path.split('/')[-2]}"
embedding_path = './embeddings/diginetica/epoch99.pt'
# embedding_path = '/mnt/e/desktop/github/SR-Predict-AO-test/DIDN+NDF/embeddings/diginetica/epoch116.pt'
os.makedirs(save_path, exist_ok=True)
pretrained_embedding = None
pretrained_embedding = torch.load(embedding_path)
pretrained_embedding = torch.nn.Parameter(pretrained_embedding)

def main():
    print('Loading data...')
    train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)
    
    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    if args.dataset_path.split('/')[-2] == 'diginetica':
        n_items = 43098
    elif args.dataset_path.split('/')[-2] in ['yoochoose1_64', 'yoochoose1_4']:
        n_items = 37484
    else:
        raise Exception('Unknown Dataset!')

    model = NARM(n_items, args.hidden_size, args.embed_dim, args.batch_size)
    if pretrained_embedding is not None:
        model.emb.weight = pretrained_embedding
        model.emb.requires_grad_(False)

    model = model.to(device)

    if args.test:
        ckpt = torch.load('latest_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        recall, mrr = validate(test_loader, model)
        print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(args.topk, recall, args.topk, mrr))
        return

    params_config = [
        {"params": (p for n, p in model.named_parameters() if 'emb' not in n)},
        {"params": model.emb.parameters(), "lr": 0.1 * args.lr}
    ]

    optimizer = optim.Adam(params_config, args.lr)
    # optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    for epoch in tqdm(range(args.epoch)):
        if epoch == 5:
            model.emb.requires_grad_(True)
            print('===== Start fine-tuning embedding layer. =====')
        # train for one epoch
        scheduler.step(epoch = epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 200)

        recall, mrr = validate(valid_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr))

        if pretrained_embedding is None:
            torch.save(model.emb.weight.cpu(), f'{save_path}/epoch{epoch}.pt')

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))
            
            if model.emb.weight.grad is not None:
                grad_l2=torch.norm(model.emb.weight.grad, p=2, dim=1)
                grad_l1=torch.norm(model.emb.weight.grad, p=1, dim=1)
            else:
                grad_l2 = torch.zeros(1)
                grad_l1 = torch.zeros(1)
            print(f'Embedding gradient norm: L2 = {torch.mean(grad_l2).item()}, L1 = {torch.mean(grad_l1).item()}')


        start = time.time()


def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = metric.evaluate(logits, target, k = args.topk)
            recalls.append(recall)
            mrrs.append(mrr)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr


if __name__ == '__main__':
    main()
