import pdb
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import BaselineModel
from dataloaders import DDIData
from utils import load_checkpoint, save_checkpoint, get_logger, setup_experiment_log

NUM_DRUGS = 4264
NUM_LABELS = 299

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--savedir',     type=str,   default='./results/ddi/',       help='directory to save results')
    parser.add_argument('--exp_name',    type=str,   default='test',                 help='Name of experiment (used in creating save location as well')
    parser.add_argument('--test_epoch',  type=int,   default=10,                     help='How often to run model on test set')
    parser.add_argument('--data_fn',          type=str,   default='./data/ddi_pairs.txt', help='location of DDI pairs text file')
    parser.add_argument('--num_workers', type=int,   default=1)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--train_pct',   type=float, default=0.8)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--embed_dim',   type=int,   default=32)
    parser.add_argument('--hid_dim',     type=int,   default=32)
    parser.add_argument('--out_dim',     type=int,   default=32)
    parser.add_argument('--cuda',       action='store_true', default=False, help='Flag to specify cuda')
    parser.add_argument('--save',       action='store_true', default=False, help='Flag to specify to save log, summary writer')
    parser.add_argument('--pin_memory', action='store_true', default=True)
    args = parser.parse_args()
    return args

def ncorrect(output, tgt):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == tgt).sum().item()
    return correct

def validate_model(dataloader, model, device):
    '''
    Returns: float, accuracy of model on the data in the given dataloader
    '''
    tot_correct = 0
    tot = 0

    with torch.no_grad():
        for xs, y in dataloader:
            xs, y = xs.to(device), y.to(device)
            ypred = model.forward(xs)
            tot_correct += ncorrect(ypred, y)
            tot += xs.shape[0]

    acc = tot_correct / tot
    return acc

def main(args):
    log_fn, swr = setup_experiment_log(args, args.savedir, args.exp_name, save=args.save)
    log = get_logger(log_fn)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    log.info(f'Starting experiment on device {device}. Saving log in: {log_fn}')

    # Set up train, test data loaders
    loader_params = {'batch_size': args.batch_size, 'num_workers': args.num_workers,
                     'pin_memory': args.pin_memory, 'shuffle': True}
    data = DDIData(args.data_fn)
    train_len = int(len(data) * args.train_pct)
    test_len = len(data) - train_len
    train_data, test_data = torch.utils.data.random_split(data,
                                                          (train_len, test_len),
                                                          torch.Generator().manual_seed(args.seed))
    train_loader = torch.utils.data.DataLoader(train_data, **loader_params)
    test_loader = torch.utils.data.DataLoader(test_data, **loader_params)

    # set up model, optimizer
    model = BaselineModel(NUM_DRUGS, args.embed_dim, args.hid_dim, NUM_LABELS)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    savedir = os.path.join(args.savedir, args.exp_name)
    checkpoint_fn = os.path.join(savedir, 'checkpoint.pth')
    model, opt, start_epoch, load_success = load_checkpoint(model, opt, log, checkpoint_fn)
    losses = [0]
    for e in range(start_epoch, start_epoch + args.epochs + 1):
        if e % args.test_epoch == 0:
            model.eval()
            val_acc = validate_model(test_loader, model, device)
            log.info('Epoch {:5d} | Last epoch train loss {:.3f} | Test acc: {:.3f}'.format(e, np.mean(losses), val_acc))
            model.train()
            if args.save:
                save_checkpoint(e, model, opt, checkpoint_fn)

        losses = []
        for xs, y in train_loader:
            opt.zero_grad()
            xs, y = xs.to(device), y.to(device)
            ypred = model.forward(xs)
            loss = criterion(ypred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

if __name__ == '__main__':
    args = get_args()
    main(args)
