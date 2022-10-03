import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class UGCL(nn.Module):
    def __init__(self, n_in, n_h):
        super(UGCL, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.sigm = nn.Sigmoid()

    def forward(self, bf, ba, num_hop, sparse):
        a_emb = self.gcn1(bf, ba, sparse)
        b_emb = self.gcn2(bf, ba, sparse)

        f = lambda x: torch.exp(x)

        n_hop_bemb = b_emb[0]
        for i in range(num_hop):
            n_hop_bemb = ba @ n_hop_bemb

        inter_sim = f(sim(a_emb[0], n_hop_bemb))
        intra_sim = f(sim(a_emb[0], a_emb[0]))
        loss = -torch.log(inter_sim.diag()/
                           (intra_sim.sum(dim=-1) - intra_sim.diag()))

        return loss.mean()

    def embed(self, seq, adj, sparse, num_hop):
        h_1 = self.gcn1(seq, adj, sparse)

        h_2 = copy.copy(h_1)[0].cpu()
        adj = adj.cpu()
        for i in range(num_hop):
            if i == 0:
                h_1 = copy.copy(h_2)

            h_2 = adj @ h_2

        h_1 = h_1.unsqueeze(0)

        h_2 = h_2.unsqueeze(0)

        # return ((h_1.cpu() + h_2)).detach()
        return h_2.detach()

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)