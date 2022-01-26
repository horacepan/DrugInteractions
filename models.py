import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv
from egnn_clean import EGNN

def _reset_params(model):
    for p in model.parameters():
        if len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid_dim, out_dim):
        super(BaselineModel, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.enc = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        self.reset_params()

    def forward(self, xs):
        xs = self.enc(xs)
        xs = xs.sum(dim=1)
        return self.dec(xs)

    def reset_params(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

class BasicGCN(torch.nn.Module):
    def __init__(self, embed_dim, hid_dim, out_dim, nlayers, dropout=0.5):
        super(BasicGCN, self).__init__()
        self.atom_encoder = nn.Embedding(11, embed_dim) # 11 different atom types here
        self.gcn_layers = nn.ModuleList(
            [GCNConv(embed_dim, hid_dim)] + \
            [GCNConv(hid_dim, hid_dim) for _ in range(nlayers - 1)]
        )
        self.fc = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout
        self.nlayers = nlayers

    def forward(self, x, edge_index, batch, edge_weight=None, **kwargs):
        x = self.atom_encoder(x)

        for gconv in self.gcn_layers:
            x = gconv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        return x

class BasicGCN2(torch.nn.Module):
    def __init__(self, embed_dim, hid_dim, out_dim, nlayers, dropout=0.5):
        super(BasicGCN2, self).__init__()
        self.atom_encoder = nn.Embedding(11, embed_dim) # 11 different atom types here
        self.gcn_layers = nn.ModuleList(
            [GraphConv(embed_dim, hid_dim)] + \
            [GraphConv(hid_dim, hid_dim) for _ in range(nlayers - 1)]
        )
        self.fc = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout
        self.nlayers = nlayers

    def forward(self, x, edge_index, batch, edge_weight=None, **kwargs):
        x = self.atom_encoder(x)

        for gconv in self.gcn_layers:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        return x

class GCNPair(nn.Module):
    def __init__(self, embed_dim, hid_dim, out_dim, nlayers=2, dropout=0.5, dec='mlp', base_gcn='BasicGCN', nopos=False):
        super(GCNPair, self).__init__()
        if base_gcn == 'BasicGCN':
            self.gcn = BasicGCN(embed_dim, hid_dim, hid_dim, nlayers, dropout)
        elif base_gcn == 'GraphConv':
            self.gcn = BasicGCN2(embed_dim, hid_dim, hid_dim, nlayers, dropout)
        elif base_gcn == 'EGNN':
            self.gcn = EGNN(embed_dim, hid_dim, hid_dim, n_layers=nlayers, nopos=nopos)
        if dec == 'mlp':
            self.dec = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, out_dim),
            )
        else:
            self.dec = nn.Linear(hid_dim, out_dim)
        _reset_params(self)

    def forward(self, batch):
        x1, edge_index1, batch1, pos1 = batch.x1, batch.edge_index1, batch.x1_batch, batch.pos1
        x2, edge_index2, batch2, pos2 = batch.x2, batch.edge_index2, batch.x2_batch, batch.pos2

        # gcn needs atom features, edge index, batch (for pooling), and potentially edge attrs, coordinates,
        g1 = self.gcn(x=x1, edge_index=edge_index1, batch=batch1, coord=pos1)
        g2 = self.gcn(x=x2, edge_index=edge_index2, batch=batch2, coord=pos2)
        gs = g1 + g2
        gs = F.relu(gs)
        output = self.dec(gs)
        return output

    def forward_all(self, x1, e1, b1, x2, e2, b2, ew1=None, ew2=None):
        g1 = self.gcn(x1, e1, b1, ew1)
        g2 = self.gcn(x2, e2, b2, ew2)
        gs = g1 + g2
        gs = F.relu(gs)
        output = self.dec(gs)
        return output

class GCNEntPair(nn.Module):
    def __init__(self, vocab_size, embed_dim, ghid_dim, ehid_dim, out_dim, nlayers=2, dropout=0.5, base_gcn='', nopos=True):
        super(GCNEntPair, self).__init__()
        if base_gcn == 'GCN':
            self.gcn = BasicGCN(embed_dim, ghid_dim, ghid_dim, nlayers, dropout)
        elif base_gcn == 'GraphConv':
            self.gcn = BasicGCN2(embed_dim, ghid_dim, ghid_dim, nlayers, dropout)
        elif base_gcn == 'EGNN':
            self.gcn = EGNN(embed_dim, ghid_dim, ghid_dim, n_layers=nlayers, nopos=nopos)
        self.enc = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, ehid_dim),
            nn.ReLU(),
            nn.Linear(ehid_dim, ehid_dim),
            nn.ReLU(),
        )
        hid_dim = ghid_dim + ehid_dim
        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        _reset_params(self)

    def forward(self, batch):
        #x1, edge_index1, edge_attr1, pos1, ent1, batch1 = batch.x, batch.edge_index, batch.edge_attr, batch.pos, batch.ent, batch.batch
        #x2, edge_index2, edge_attr2, pos2, ent2, batch2 = batch.x2, batch.edge_index2, batch.edge_attr2, batch.pos2, batch.ent2, batch.x2_batch
        x1, edge_index1, ent1, batch1, pos1 = batch.x1, batch.edge_index1, batch.ent1, batch.x1_batch, batch.pos1
        x2, edge_index2, ent2, batch2, pos2 = batch.x2, batch.edge_index2, batch.ent2, batch.x2_batch, batch.pos2

        g1 = self.gcn(x=x1, edge_index=edge_index1, batch=batch1, coord=pos1)
        g2 = self.gcn(x=x2, edge_index=edge_index2, batch=batch2, coord=pos2)
        e1 = self.enc(ent1)
        e2 = self.enc(ent2)

        eg1 = torch.hstack([g1, e1])
        eg2 = torch.hstack([g2, e2])
        egs = eg1 + eg2
        egs = F.relu(egs)
        output = self.dec(egs)
        return output

class EntNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid_dim, out_dim):
        super(EntNet, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.enc = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        _reset_params(self)

    def forward(self, batch):
        xs = torch.stack([batch.ent1, batch.ent2], dim=1)
        xs = self.enc(xs)
        xs = xs.sum(dim=1)
        xs = F.relu(xs)
        return self.dec(xs)

class MorganFPNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MorganFPNet, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, drug1, drug2):
        d1 = self.enc(drug1)
        d2 = self.enc(drug2)
        d = d1 + d2
        d = self.dec(d)
        return d

if __name__ == '__main__':
    net = BaselineModel(100, 32, 32, 300)
    x = torch.randint(0, 100, size=(37, 2))
    ys = net(x)
    print(ys.shape)
