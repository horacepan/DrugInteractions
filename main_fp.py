import pdb
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset
from parse_structure import parse_structure
from models import MorganFPNet
from dataloaders import FingerPrintDataset, SmilesFPDataset
from utils import load_checkpoint, save_checkpoint, get_logger, setup_experiment_log, check_memory
from main_graph import ncorrect, num_params

def _validate_model(dataloader, model, device, log):
    '''
    Returns: float, accuracy of model on the data in the given dataloader
    '''
    tot_correct = 0
    tot = 0
    with torch.no_grad():
        for batch in dataloader:
            d1 = batch[0].float().to(device)
            d2 = batch[1].float().to(device)
            target = torch.LongTensor(batch[2])
            # batch = batch.to(device)
            #ypred = model.forward(batch)
            ypred = model.forward(d1, d2)
            tot_correct += ncorrect(ypred, target)
            tot += len(batch[2])

    acc = tot_correct / tot
    return acc

def main(args):
    log_fn, swr = setup_experiment_log(args, args.savedir, args.exp_name, save=args.save)
    log = get_logger(log_fn)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    log.info(f'Starting experiment on device {device}. Saving log in: {log_fn}')
    log.info(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if 'neg' in args.data_fn:
        dataset = SmilesFPDataset(args.data_fn, no_neg=args.no_neg)
    else:
        dataset = FingerPrintDataset(args.data_fn, args.pkl_fn, {"radius": args.fp_radius})
    log.info("Dataset len: {}".format(len(dataset)))
    train_len = int(len(dataset) * args.train_pct)
    test_len = len(dataset) - train_len
    train_data, test_data = torch.utils.data.random_split(dataset,
                                                          (train_len, test_len),
                                                          torch.Generator().manual_seed(args.seed))
    loader_params = {'batch_size': args.batch_size, 'shuffle': True}
    train_loader = DataLoader(dataset=train_data, **loader_params)
    test_loader = DataLoader(dataset=test_data, **loader_params)
    log.info("Train set: {} | Test set: {}".format(len(train_loader), len(test_loader)))

    model = MorganFPNet(2048, args.hid_dim, 299)
    model = model.to(device)
    log.info("Made model")
    check_memory()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    log.info(f'Running Model: {model.__class__} | num params: {num_params(model)}')
    check_memory()

    savedir = os.path.join(args.savedir, args.exp_name)
    checkpoint_fn = os.path.join(savedir, 'checkpoint.pth')
    model, opt, start_epoch, load_success = load_checkpoint(model, opt, log, checkpoint_fn)
    losses = [0]
    loss_hist = []

    for e in range(start_epoch, start_epoch + args.epochs + 1):
        if e % args.test_epoch == 0:
            model.eval();
            val_acc = _validate_model(test_loader, model, device, log)
            model.train()
            if args.save:
                save_checkpoint(e, model, opt, checkpoint_fn)
                log.info('Epoch {:5d} | Last epoch train loss {:.3f} | Test acc: {:.3f}'.format(e, np.mean(losses), val_acc))
            else:
                log.info('Epoch {:5d} | Last epoch train loss {:.3f} | Test acc: {:.3f} | -'.format(e, np.mean(losses), val_acc))

        losses = []
        for batch in train_loader:
            opt.zero_grad()
            #batch = batch.to(device)
            d1 = batch[0].float().to(device)
            d2 = batch[1].float().to(device)
            target = torch.LongTensor(batch[2]).to(device)
            ypred = model.forward(d1, d2)
            loss = criterion(ypred, target)
            loss.backward()
            opt.step()

            losses.append(loss.item())
        loss_hist.append(np.mean(losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--savedir',     type=str,   default='./results/ddi/',       help='directory to save results')
    parser.add_argument('--exp_name',    type=str,   default='test',                 help='Name of experiment (used in creating save location as well')
    parser.add_argument('--test_epoch',  type=int,   default=1,                      help='How often to run model on test set')
    #parser.add_argument('--data_fn',     type=str,   default='./data/ddi_pairs.txt', help='location of DDI pairs text file')
    parser.add_argument('--data_fn',     type=str,   default='./data/ddi_pos_neg_uniq_smiles.tsv', help='location of DDI pairs text file')
    parser.add_argument('--pkl_fn',      type=str,   default='./data/db_smiles.pkl', help='location of DBID -> smiles dict pickle file')
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--train_pct',   type=float, default=0.8)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--hid_dim',     type=int,   default=256)
    parser.add_argument('--fp_radius',   type=int,   default=2)
    parser.add_argument('--no_neg',     action='store_true', default=False, help='whether or not to use neg examples')
    parser.add_argument('--cuda',       action='store_true', default=False, help='Flag to specify cuda')
    parser.add_argument('--save',       action='store_true', default=False, help='Flag to specify to save log, summary writer')
    args = parser.parse_args()

    main(args)
