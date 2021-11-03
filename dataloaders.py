import pdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class DDIData(Dataset):
    def __init__(self, fn, nrows=None):
        self._col_names = ['Drug1', 'Drug2', 'ID']
        self._df = pd.read_csv(fn, sep='\t', nrows=nrows)

        drugs, labels = self._load_df_fast(self._df)
        self.drugs = torch.from_numpy(drugs)
        self.labels = torch.from_numpy(labels)

    def _load_df_fast(self, df):
        '''
        df: pandas dataframe of the the DDI pairs and interaction
        Returns: tuple of (np.array, np.array)
            containing the drug pairs (mapped to int ids), interaction label
        '''
        unique_drugs = set(df['Drug1'].unique())
        unique_drugs.update(df['Drug2'].unique())
        unique_drugs = sorted(unique_drugs)
        unique_labels = sorted(df['ID'].unique())

        dmap = {d: idx for idx, d in enumerate(unique_drugs)}
        lmap = {l: idx for idx, l in enumerate(unique_labels)}

        d1 = df['Drug1'].apply(lambda x: dmap[x]).to_numpy()
        d2 = df['Drug2'].apply(lambda x: dmap[x]).to_numpy()
        labels = df['ID'].apply(lambda x: lmap[x]).to_numpy()
        drugs = np.column_stack((d1, d2))
        return drugs, labels

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        drugs = self.drugs[idx]
        label = self.labels[idx]
        return drugs, label

if __name__ == '__main__':
    fn = './data/ddi_pairs.txt'
    dataset = DDIData(fn)
