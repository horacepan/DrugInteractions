import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid_dim, out_dim):
        super(BaselineModel, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = hid_dim
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
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, batch):
        x = self.atom_encoder(x)

        for gconv in self.gcn_layers:
            x = gconv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        return x

class GCNPair(nn.Module):
    def __init__(self, embed_dim, hid_dim, out_dim, nlayers=2, dropout=0.5):
        super(GCNPair, self).__init__()
        self.gcn = BasicGCN(embed_dim, hid_dim, hid_dim, nlayers, dropout)
        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, batch):
        #x1, edge_index1, edge_attr1, pos1, ent1, batch1 = batch.x, batch.edge_index, batch.edge_attr, batch.pos, batch.ent, batch.batch
        #x2, edge_index2, edge_attr2, pos2, ent2, batch2 = batch.x2, batch.edge_index2, batch.edge_attr2, batch.pos2, batch.ent2, batch.x2_batch
        x1, edge_index1, batch1 = batch.x1, batch.edge_index1, batch.x1_batch
        x2, edge_index2, batch2 = batch.x2, batch.edge_index2, batch.x2_batch

        g1 = self.gcn(x1, edge_index1, batch1)
        g2 = self.gcn(x2, edge_index2, batch2)
        gs = g1 + g2
        output = self.dec(gs)
        return output

class GCNEntPair(nn.Module):
    def __init__(self, vocab_size, embed_dim, ghid_dim, ehid_dim, out_dim, nlayers=2, dropout=0.5):
        super(GCNEntPair, self).__init__()
        self.gcn = BasicGCN(embed_dim, ghid_dim, ghid_dim, nlayers, dropout)
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
    def forward(self, batch):
        #x1, edge_index1, edge_attr1, pos1, ent1, batch1 = batch.x, batch.edge_index, batch.edge_attr, batch.pos, batch.ent, batch.batch
        #x2, edge_index2, edge_attr2, pos2, ent2, batch2 = batch.x2, batch.edge_index2, batch.edge_attr2, batch.pos2, batch.ent2, batch.x2_batch
        x1, edge_index1, ent1, batch1 = batch.x1, batch.edge_index1, batch.ent1, batch.x1_batch
        x2, edge_index2, ent2, batch2 = batch.x2, batch.edge_index2, batch.ent2, batch.x2_batch

        g1 = self.gcn(x1, edge_index1, batch1)
        g2 = self.gcn(x2, edge_index2, batch2)
        e1 = self.enc(ent1)
        e2 = self.enc(ent2)

        eg1 = torch.hstack([g1, e1])
        eg2 = torch.hstack([g2, e2])
        egs = eg1 + eg2
        output = self.dec(egs)
        return output

if __name__ == '__main__':
    net = BaselineModel(100, 32, 32, 300)
    x = torch.randint(0, 100, size=(37, 2))
    ys = net(x)
    print(ys.shape)
