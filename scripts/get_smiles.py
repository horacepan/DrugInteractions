import time
import pdb
import pickle
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd
from multiprocessing import Pool

CID_TAG = 'PC-CompoundType_id_cid'

def unique_dbids(nrows=None):
    fn = '../data/ddi_pairs.txt'
    cols = ['Drug1', 'Drug2']
    df = pd.read_csv(fn, nrows=nrows, sep='\t')
    uniques = pd.unique(df[cols].values.ravel('K'))
    uniques.sort()
    return uniques

def dburl(dbid):
    fmt = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/drugbank/{}/XML'
    return fmt.format(dbid)

def curl(cid):
    fmt = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/XML'
    return fmt.format(cid)

def get_cid_from_dbid(dbid):
    url = dburl(dbid)
    doc = requests.get(url)
    soup = BeautifulSoup(doc.content, "lxml-xml")
    res = soup.find(CID_TAG)

    if not res is None:
        cid = res.text
        return cid

def get_smiles_from_cid(cid):
    url = curl(cid)
    doc = requests.get(url)
    soup = BeautifulSoup(doc.content, "lxml-xml")

    res = soup.find_all('PC-InfoData_value_sval')
    if len(res) < 3:
        smiles = ''
    else:
        smiles = res[-3].text
    return smiles

def get_smiles_from_dbid(dbid):
    cid = get_cid_from_dbid(dbid)
    if cid is None:
        return ''

    smiles = get_smiles_from_cid(cid)
    return smiles

def sample(x):
    return x

def main_pool(nrows, ncpu, chunksize):
    dbids = unique_dbids(nrows)
    print(f'Num drugs: {len(dbids)}')

    with Pool(processes=ncpu) as pool:
        res = pool.map(get_smiles_from_dbid, dbids, chunksize=chunksize)

def main(nrows=None):
    smiles = []
    dbids = unique_dbids(nrows)
    smiles_dict = {}

    for dbid in tqdm(dbids):
        s = get_smiles_from_dbid(dbid)
        smiles.append(s)
        if s:
            smiles_dict[dbid] = s

    print("Num unique db ids: {}".format(len(dbids)))
    print("Drugs with smiles: {}".format(len(smiles_dict)))
    f = open('../data/db_smiles.pkl', 'wb')
    pickle.dump(smiles_dict, f)

    print("Done pickling")
    df = pd.DataFrame({"DBID": dbids, "SMILES": smiles})
    df.to_csv("drug_smiles.csv")
    return smiles_dict

if __name__ == '__main__':
    nrows = None
    st = time.time()
    main(nrows)
    end = time.time()
    print("Elapsed w/o pool: {:.2f}s".format(end - st))
