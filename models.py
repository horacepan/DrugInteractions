import torch
import torch.nn as nn

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

    def forward(self, xs):
        xs = self.enc(xs)
        xs = xs.sum(dim=1)
        return self.dec(xs)

if __name__ == '__main__':
    net = BaselineModel(100, 32, 32, 300)
    x = torch.randint(0, 100, size=(37, 2))
    ys = net(x)
    print(ys.shape)
