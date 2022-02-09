import time
import pickle
import pdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as tDataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from parse_structure import parse_structure
from models import MorganFPNet

from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors

from utils import check_memory

def _get_morgan_fp(smile_str, fp_param_dict):
    mol = Chem.MolFromSmiles(smile_str)
    fp = np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, **fp_param_dict))
    return fp

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
                       x2=None, edge_index2=None, edge_attr2=None, pos2=None, ent2=None, target=None, d1=None, d2=None):
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
        self.d1 = d1
        self.d2 = d2

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
        self.smiles = pickle.load(open('./data/db_to_smiles.pkl', 'rb'))

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
        pair_data = PairData(x1=data1.x, edge_index1=data1.edge_index, edge_attr1=data1.edge_attr, pos1=data1.pos, ent1=torch.tensor([ent1],),
                             x2=data2.x, edge_index2=data2.edge_index, edge_attr2=data2.edge_attr, pos2=data2.pos, ent2=torch.tensor([ent2]),
                             target=torch.tensor([label]), d1=drug1, d2=drug2)
        return pair_data

class FingerPrintDataset(Dataset):
    def __init__(self, fn, pkl, fp_param_dict, nrows=None):
        self._db_to_smiles = pickle.load(open(pkl, 'rb'))
        self._fps = self._smiles(self._db_to_smiles, fp_param_dict) #{db: torch.FloatTensor(_get_morgan_fp(s, fp_param_dict)) for db, s in self._db_to_smiles.items()}
        self._df = _filtered_df(pd.read_csv(fn, sep='\t', nrows=nrows), set(self._fps))
        self.fp_params = fp_param_dict
        drugs, labels, dmap, lmap = _get_entities_labels(self._df)
        self.labels = labels
        self.drugs = drugs

    def _smiles(self, db_smiles, fp_params):
        d = {}
        for db, s in db_smiles.items():
            try:
                d[db] = _get_morgan_fp(s, fp_params)
            except:
                continue
        return d

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        drug_row = self._df.iloc[idx]
        d1, d2, _ = drug_row
        fp1 = torch.FloatTensor(self._fps[d1])
        fp2 = torch.FloatTensor(self._fps[d2])
        return fp1, fp2, self.labels[idx]

class SmilesFPDataset(Dataset):
    def __init__(self, fn, nrows=None, radius=2, no_neg=False):
        self.radius = radius
        self._df = pd.read_csv(fn, sep='\t')
        self._fp_cache = self._cache_smiles_fps(self._df)
        self._df = self._filter_df(self._df, set(self._fp_cache.keys()))
        self._label_map = {}

        if no_neg:
            self._df = self._df[self._df['ID'] != -1]

        labels = self._df['ID'].unique()
        for idx, l in np.ndenumerate(labels):
            self._label_map[l] = idx[0]

    def _cache_smiles_fps(self, df):
        s1 = df[df.columns[0]].unique()
        s2 = df[df.columns[1]].unique()
        unique_smiles = list(set(np.concatenate([s1, s2])))
        cache = {}

        for s in unique_smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                fp = np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=self.radius))
                cache[s] = fp

        return cache

    def _filter_df(self, df, val_set):
        df1in = df[df.columns[0]].isin(val_set)
        df2in = df[df.columns[1]].isin(val_set)
        return df[df1in & df2in]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        d1, d2, l = self._df.iloc[idx]
        label = self._label_map[l]
        fp1 = self._fp_cache[d1]
        fp2 = self._fp_cache[d2]
        return fp1, fp2, label

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fn = './data/ddi_pairs.txt'
    struc_fn = './data/3d_struc.csv'
    pkl = './data/db_smiles.pkl'
    params = {'radius': 2}
    #dataset = FingerPrintDataset(fn, pkl, params)

    fn = './data/ddi_pos_neg_uniq_smiles.tsv'
    dataset = SmilesFPDataset(fn)
    dfd = tDataLoader(dataset, batch_size=32)
    check_memory()
    print(len(dataset))

    net = MorganFPNet(2048, 256, 299).to(device)
    check_memory()
    for batch in dfd:
        a, b, c = batch
        a = a.to(device)
        b = b.to(device)
        check_memory()
        pdb.set_trace()
    #dld = DataLoader(d2, follow_batch=d2.follow_batch, batch_size=32)
    #batch = next(iter(dld))
    #end = time.time()
    #print('Elapsed: {:.2f}s'.format(end - st))
