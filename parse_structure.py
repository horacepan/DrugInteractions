import time
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

Bond = namedtuple("Bond", ['fro','to', 'type'])
#ATOMIC_NUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 36, 'I': 53}
ATOM_TYPES =  {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Si':  5, 'P':  6, 'S':  7, 'Cl':  8, 'Br':  9, 'I': 10}

def _get_edges(bonds):
    '''
    bonds: list of Bond named tuples
    Returns: tuple of (list of ints, list of ints, list of ints)
        1st list of ints: int id of head of bond
        2nd list of ints: int id of tail of bond
        3rd list of ints: bond types
    Note: input bonds list is 1-indexed but the return is 0-indexed
    '''
    edges_from, edges_to = [], []
    edge_types = []

    for bond in bonds:
        # sub 1 to get 0-indexed values
        a1 = bond.fro - 1
        a2 = bond.to - 1
        bond_type = bond.type

        # add in both directions to make undirected
        edges_from.append(a1)
        edges_from.append(a2)
        edges_to.append(a2)
        edges_to.append(a1)
        edge_types.append(bond_type)
        edge_types.append(bond_type)

    return edges_from, edges_to, edge_types

def _get_atoms(atom_dict):
    '''
    atom_dict: dict mapping Atom string -> tuple of floats representing the atom's coordinates
    Returns: tuple of atom types(list of ints) and coordinates (numpy array)
    '''
    atom_types = []
    coords = np.zeros((len(atom_dict), 3))
    idx = 0
    graph_sizes = []

    for atom_str, xyz in atom_dict.items():
        if type(atom_str) != str or 'Atom' not in atom_str:
            pdb.set_trace()
            continue
        coords[idx] = xyz
        idx += 1

        # atom_str looks like: 'Atom(id, c)'
        popen_idx =  atom_str.index('(') # index of open parens
        comma_idx =  atom_str.index(',')
        pclose_idx = atom_str.index(')')

        atom_id = int(atom_str[popen_idx + 1: comma_idx]) - 1 # sub 1 to get 0 index
        atom_type_str = atom_str[comma_idx + 1: pclose_idx].strip() # string rep of atom type
        atom_types.append(atom_type_str)

    return atom_types, coords

def make_data(struc_str, onehot=False):
    '''
    struc_str: string that holds the atom, coordinate, and bond information
    Returns: torch_geometric.data.Data object
    '''
    atom_dict = eval(struc_str)
    bond_lst = atom_dict.pop('bonds')
    bond_st, bond_end, bond_types = _get_edges(bond_lst)
    atoms, coords = _get_atoms(atom_dict)
    atoms = [ATOM_TYPES[a] for a in atoms]

    if onehot:
        atom_features = F.one_hot(torch.tensor(atoms), num_classes=len(ATOM_TYPES))
        edge_attr = F.one_hot(torch.tensor(bond_types), num_classes=3)
    else:
        atom_features = torch.tensor(atoms)
        edge_attr = torch.tensor(bond_types)
    edge_index = torch.tensor([bond_st, bond_end], dtype=torch.long)
    pos = torch.tensor(coords)
    data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    return data

def parse_structure(fn, nrows=None, onehot=False):
    '''
    fn: string of location of the 3d_structure.csv file
    Returns: list of torch_geometric.data.Data objects
    '''
    df = pd.read_csv(fn, nrows=nrows)
    drug_map = {}
    data_lst = []
    max_atoms = max_edges = 0

    for row_idx, row in tqdm(df.iterrows()):
        drug = row['drugbank_id']
        struc_str = row['structure']
        atom_dict = eval(struc_str)
        bond_lst = atom_dict.pop('bonds')

        bond_st, bond_end, bond_types = _get_edges(bond_lst)
        atoms, coords = _get_atoms(atom_dict)
        atoms = [ATOM_TYPES[a] for a in atoms]

        if onehot:
            atom_features = F.one_hot(torch.tensor(atoms), num_classes=len(ATOM_TYPES))
            edge_attr = F.one_hot(torch.tensor(bond_types), num_classes=3)
        else:
            atom_features = torch.tensor(atoms)
            edge_attr = torch.tensor(bond_types)

        edge_index = torch.tensor([bond_st, bond_end], dtype=torch.long)
        pos = torch.tensor(coords)
        data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        drug_map[drug] = data
        data_lst.append(data)

    return data_lst, drug_map

if __name__ == '__main__':
    fn = './data/3d_struc.csv' # change location to wherever the file lives
    data_lst, drug_map = parse_structure(fn)
