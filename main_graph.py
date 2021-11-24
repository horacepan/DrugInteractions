import time
import pdb
from torch_geometric.data import Data, DataLoader
from dataloaders import *
from models import BaselineModel, GCNPair, GCNEntPair

def main():
    fn = './data/ddi_pairs.txt'
    struc_fn = './data/3d_struc.csv'
    nrows =  10000
    vocab_size = 4264
    nclasses = 299
    embed_dim = 32
    hid_dim = 64
    out_dim = nclasses

    dataset = DDIGraphDataset(fn, struc_fn, nrows=nrows)
    dataloader = DataLoader(dataset, follow_batch=dataset.follow_batch, batch_size=32)
    diter = iter(dataloader)

    mg = GCNPair(embed_dim, hid_dim, out_dim)
    me = GCNEntPair(vocab_size, embed_dim, hid_dim // 2, hid_dim // 2, out_dim)

    batch = next(diter)
    mg_res = mg(batch)
    m2_res = me(batch)
    pdb.set_trace()

if __name__ == '__main__':
    main()
