from utils import load_checkpoint, save_checkpoint, get_logger
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models import BaselineModel
from dataloaders import DDIData
NUM_DRUGS = 4264

def get_model(args, **kwargs):
    model = BaselineModel(kwargs['num_drugs'], args.embed_dim, args.hid_dim, args.out_dim)
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savedir', type=str, default='./results/ddi/')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=32)
    args = parser.parse_args()
    return args

def main(args):
    log, swr = setup_experiment()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    # Set up train, test data loaders
    loader_params = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': args.pin_memory, 'shuffle': True}
    data = DDIData(args.fn)
    train_len = int(len(data) * args.train_pct)
    test_len = len(data) - train_len
    train_data, test_data = torch.utils.data.random_split(data,
                                                          (train_len, test_len),
                                                          torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_data, **loader_params)
    test_loader = DataLoader(test_data, **loader_params)

    # set up model, optimizer
    model = get_model(args, num_drugs=NUM_DRUGS)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(start_epoch, start_epoch + args.epochs + 1):
        for xs, y in train_loader:
            opt.zero_grad()
            xs, y = xs.to(device), y.to(device)
            ypred = model.forward(xs)
            loss = criterion(ypred, y)
            opt.step()

if __name__ == '__main__':
    args = get_args()
    main(args)
