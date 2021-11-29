def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import argparse
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from dataloaders import DDIData, DDIGraphDataset
from models import BaselineModel, GCNPair, GCNEntPair, EntNet
from utils import load_checkpoint, save_checkpoint, get_logger, setup_experiment_log, check_memory

VOCAB_SIZE = 4264 # number of unique drugs
NUM_LABELS = 299

def ncorrect(output, tgt):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == tgt).sum().item()
    return correct

def _validate_model(dataloader, model, device):
    '''
    Returns: float, accuracy of model on the data in the given dataloader
    '''
    tot_correct = 0
    tot = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            ypred = model.forward(batch)
            tot_correct += ncorrect(ypred, batch.target)
            tot += batch.target.numel()

    acc = tot_correct / tot
    return acc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--savedir',     type=str,   default='./results/ddi/',       help='directory to save results')
    parser.add_argument('--exp_name',    type=str,   default='test',                 help='Name of experiment (used in creating save location as well')
    parser.add_argument('--test_epoch',  type=int,   default=1,                     help='How often to run model on test set')
    parser.add_argument('--data_fn',     type=str,   default='./data/ddi_pairs.txt', help='location of DDI pairs text file')
    parser.add_argument('--struc_fn',     type=str,   default='./data/3d_struc.csv', help='location of DDI pairs text file')
    parser.add_argument('--model_str',   type=str,   default='GCNPair', help='location of DDI pairs text file')
    parser.add_argument('--num_workers', type=int,   default=1)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--train_pct',   type=float, default=0.8)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--embed_dim',   type=int,   default=32)
    parser.add_argument('--dec',         type=str,   default='mlp')
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--nlayers',     type=int,   default=2)
    parser.add_argument('--hid_dim',     type=int,   default=32)
    parser.add_argument('--out_dim',     type=int,   default=32)
    parser.add_argument('--nrows',       type=int,   default=None)
    parser.add_argument('--cuda',       action='store_true', default=False, help='Flag to specify cuda')
    parser.add_argument('--debug',      action='store_true', default=False)
    parser.add_argument('--save',       action='store_true', default=False, help='Flag to specify to save log, summary writer')
    parser.add_argument('--pin_memory', action='store_true', default=True)
    args = parser.parse_args()
    return args

def _get_model(args):
    if args.model_str == 'GCNPair':
        model = GCNPair(args.embed_dim, args.hid_dim, NUM_LABELS, nlayers=args.nlayers, dec=args.dec, dropout=args.dropout)
    elif args.model_str == 'GCNEntPair':
        model = GCNEntPair(VOCAB_SIZE, args.embed_dim, args.hid_dim // 2, args.hid_dim // 2, NUM_LABELS)
    elif args.model_str == 'EntNet':
        model = EntNet(VOCAB_SIZE, args.embed_dim, args.hid_dim, NUM_LABELS)
    return model

def main(args):
    log_fn, swr = setup_experiment_log(args, args.savedir, args.exp_name, save=args.save)
    log = get_logger(log_fn)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    log.info(f'Starting experiment on device {device}. Saving log in: {log_fn}')

    dataset = DDIGraphDataset(args.data_fn, args.struc_fn, nrows=args.nrows)
    train_len = int(len(dataset) * args.train_pct)
    test_len = len(dataset) - train_len
    train_data, test_data = torch.utils.data.random_split(dataset,
                                                          (train_len, test_len),
                                                          torch.Generator().manual_seed(args.seed))
    log.info(f'Done loading data')
    check_memory()

    loader_params = {'batch_size': args.batch_size, 'num_workers': args.num_workers,
                     'pin_memory': args.pin_memory, 'shuffle': True}
    train_loader = DataLoader(train_data, follow_batch=dataset.follow_batch, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, follow_batch=dataset.follow_batch, batch_size=args.batch_size)

    if args.debug:
        log.info('debugging, train length: {}'.format(len(test_data)))
        test_loader = train_loader

    model = _get_model(args)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    log.info(f'Running Model: {args.model_str}')

    savedir = os.path.join(args.savedir, args.exp_name)
    checkpoint_fn = os.path.join(savedir, 'checkpoint.pth')
    model, opt, start_epoch, load_success = load_checkpoint(model, opt, log, checkpoint_fn)
    losses = [0]

    for e in range(start_epoch, start_epoch + args.epochs + 1):
        if e % args.test_epoch == 0 :
            model.eval()
            val_acc = _validate_model(test_loader, model, device)
            log.info('Epoch {:5d} | Last epoch train loss {:.3f} | Test acc: {:.3f}'.format(e, np.mean(losses), val_acc))
            model.train()
            if args.save:
                save_checkpoint(e, model, opt, checkpoint_fn)

        losses = []
        for batch in train_loader:
            opt.zero_grad()
            batch = batch.to(device)
            ypred = model.forward(batch)
            loss = criterion(ypred, batch.target)
            loss.backward()
            opt.step()
            losses.append(loss.item())

if __name__ == '__main__':
    args = get_args()
    main(args)
