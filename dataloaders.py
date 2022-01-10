import time
st = time.time()
import pdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data#, DataLoader
from parse_structure import parse_structure

def _get_entities_labels(df, nrows=None):
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
    return drugs, labels, dmap, lmap

def _filtered_df(df, all_drugs):
    df1in = df['Drug1'].isin(all_drugs)
    df2in = df['Drug2'].isin(all_drugs)
    return df[df1in & df2in]

class PairData(Data):
    def __init__(self, x1=None, edge_index1=None, edge_attr1=None, pos1=None, ent1=None,
                       x2=None, edge_index2=None, edge_attr2=None, pos2=None, ent2=None, target=None):
        super().__init__()
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.x1 = x1
        self.x2 = x2
        self.edge_attr1 = edge_attr1
        self.edge_attr2 = edge_attr2
        self.ent1 = ent1
        self.ent2 = ent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.target = target

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class DDIData(Dataset):
    def __init__(self, fn, nrows=None):
        self._col_names = ['Drug1', 'Drug2', 'ID']
        self._df = pd.read_csv(fn, sep='\t', nrows=nrows)

        drugs, labels, dmap, lmap = _get_entities_labels(self._df)
        self.drugs = torch.from_numpy(drugs)
        self.labels = torch.from_numpy(labels)
        self.id_to_drug = {i: drug_id for drug_id, i in dmap.items()}
        self.id_to_label = {i: label for label, i in dmap.items()}

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        drugs = self.drugs[idx]
        label = self.labels[idx]
        return drugs, label

class DDIGraphDataset(Dataset):
    def __init__(self, fn, struc_fn, nrows=None):
        self._drug_structs = parse_structure(struc_fn, nrows, onehot=False)[1] # dont need drug list
        self._df = _filtered_df(pd.read_csv(fn, sep='\t', nrows=nrows), self._drug_structs)
        print('post filter length df:', len(self._df))

        drugs, labels, dmap, lmap = _get_entities_labels(self._df)
        self.drugs = drugs
        self.labels = labels
        self.id_to_drug = {i: drug_id for drug_id, i in dmap.items()}
        self.id_to_label = {i: label for label, i in dmap.items()}


    @property
    def follow_batch(self):
        return ['x1', 'x2']

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        drug_row = self._df.iloc[idx]
        label = self.labels[idx]

        ent1, ent2 = self.drugs[idx]
        drug1 = drug_row['Drug1']
        data1 = self._drug_structs[drug1]
        drug2 = drug_row['Drug2']
        data2 = self._drug_structs[drug2]
        pair_data = PairData(x1=data1.x, edge_index1=data1.edge_index, edge_attr1=data1.edge_attr, pos1=data1.pos, ent1=torch.tensor([ent1]),
                             x2=data2.x, edge_index2=data2.edge_index, edge_attr2=data2.edge_attr, pos2=data2.pos, ent2=torch.tensor([ent2]),
                             target=torch.tensor([label]))
        return pair_data

if __name__ == '__main__':
    fn = './data/ddi_pairs.txt'
    struc_fn = './data/3d_struc.csv'
    dataset = DDIData(fn, nrows=100)
    d2 = DDIGraphDataset(fn, struc_fn)
    #dld = DataLoader(d2, follow_batch=d2.follow_batch, batch_size=32)
    #batch = next(iter(dld))
    #end = time.time()
    #print('Elapsed: {:.2f}s'.format(end - st))
