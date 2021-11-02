import pdb
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class DDIData(Dataset):
    def __init__(self, fn):
        self._col_names = ['Drug1', 'Drug2', 'ID']
        self.df = pd.read_csv(fn, sep='\t')
        self._rev_map = {} # drug name -> id

    def _load_df(self, df):
        drugs = set()
        labels = set()

        for d1, d2, _id in df.iterrows():
            drugs.add(d1)
            drugs.add(d2)
            labels.add(_id)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d1, d2, _id = self.df.iloc[idx]
        return d1, d2, _id

if __name__ == '__main__':
    fn = './data/ddi_pairs.txt'
    dataset = DDIData(fn)
    pdb.set_trace()
