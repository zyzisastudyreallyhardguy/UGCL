import os
import argparse
import sys
import logging
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor
from torch.nn.functional import normalize
from dataset import load
import torch.nn.functional as F
from torch_geometric.seed import seed_everything
from sklearn.preprocessing import normalize
import copy

def test_evaluator(model, adj, idx_train, idx_test, features, labels, hid_units, nb_classes, xent, cuda_no, num_hop):
    features = torch.FloatTensor(features).cuda(cuda_no)
    adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj)).cuda(cuda_no)

    embeds = model.embed(features, adj, num_hop, True)/2

    train_embs = embeds[0, idx_train]

    train_lbls = labels[idx_train]
    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0
    # for _ in range(50):
    log = LogReg(hid_units, nb_classes).cuda(cuda_no)
    opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
    # log.cuda()
    for _ in range(300):
        log.train()
        opt.zero_grad()
        logits = log(train_embs)

        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(embeds)
    preds = logits.squeeze(0)
    labels = labels

    acc = torch.sum(preds[idx_test].argmax(1) == labels[idx_test]).float() / idx_test.shape[0]
    accs.append(acc * 100)
    accs = torch.tensor(accs)
    return accs.mean().item()

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

class Model_barlow(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model_barlow, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.sigm = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(n_h, affine = False)

    def forward(self, bf, ba, num_hop, sparse):
        a_emb = self.gcn1(bf, ba, sparse)
        b_emb = self.gcn2(bf, ba, sparse)

        f = lambda x: torch.exp(x)

        n_hop_bemb = b_emb[0]
        for i in range(num_hop):
            n_hop_bemb += ba[0] @ n_hop_bemb

        inter_sim = f(sim(a_emb[0], n_hop_bemb))
        intra_sim = f(sim(a_emb[0], a_emb[0]))
        loss = -torch.log(inter_sim.diag()/
                           (intra_sim.sum(dim=-1) - intra_sim.diag()))

        return loss.mean()

    def embed(self, seq, adj,  num_hop, sparse):
        h_1 = self.gcn1(seq, adj, sparse)

        h_2 = copy.copy(h_1)[0]
        for i in range(num_hop):
            h_2 = adj @ h_2

        h_2 = h_2.unsqueeze(0)
        return (h_1 + h_2).detach()

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def train(dataset, epochs, patience, lr, l2_coef, hid_dim, sample_size, num_hop, test_conv_num, cuda_no, verbose=True):
    # parameter setting portal
    nb_epochs = epochs
    patience = patience
    batch_size = 1
    lr = lr
    l2_coef = l2_coef
    hid_units = hid_dim
    sample_size = sample_size
    sparse = False

    parameter_dict = {}
    parameter_dict['dataset'] = dataset
    parameter_dict['nb_epochs'] = nb_epochs
    parameter_dict['patience'] = patience
    parameter_dict['lr'] = lr
    parameter_dict['l2_coef'] = l2_coef
    parameter_dict['hid_units'] = hid_units
    parameter_dict['sample_size'] = sample_size
    parameter_dict['n'] = num_hop

    logger.info('parameters: {}'.format(parameter_dict))

    adj, features, labels, idx_train, idx_val, idx_test = load(dataset)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    model = Model_barlow(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    xent = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    cnt_wait = 0
    best = 1e9

    best_acc = 0

    for epoch in range(nb_epochs):
        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(adj[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf[0]).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
        else:
            ba = torch.FloatTensor(ba)

        bf = torch.FloatTensor(bf)

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()

        model.train()
        optimiser.zero_grad()

        loss = model(bf, ba, num_hop, sparse)

        loss.backward()
        optimiser.step()

        if (epoch == (args.epoch - 1)) and verbose:
            model.eval()
            acc = test_evaluator(model, adj, idx_train, idx_test, features, labels, hid_units, nb_classes,
                                 xent, cuda_no, num_hop)
            print(acc)

        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), './checkpoints/{}_{}_{}_{}_{}_{}_{}.pkl'.format(dataset,
                                                                                         patience,
                                                                                         lr,
                                                                                         l2_coef,
                                                                                         hid_dim,
                                                                                         batch_size,
                                                                                         sample_size))
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            # if verbose:
            #     logger.info('Early stopping!')
            #     logger.info('The best test accuracy is : {}'.format(best_acc))
            break

    return acc

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser('UGCL')
    parser.add_argument('--data', type=str, default='cora', help='Dataset name: cora, citeseer, pubmed')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Patience')
    parser.add_argument('--lr', type=float, default=0.0001, help='Patience')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--n', type=int, default=7, help='n-th Graph Power')
    parser.add_argument('--hidden', type=int, default=4096, help='Hidden dim')
    parser.add_argument('--sample_size', type=int, default=2000, help='Sample size')
    parser.add_argument('--sparse', action='store_true', help='Whether to use sparse tensors')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)


    seed = 2022
    seed_everything(seed)
    print('seed_numer' + str(seed))

    torch.cuda.set_device(args.cuda)
    dataset = args.data
    n_runs = args.runs
    epochs = args.epoch
    patience = args.patience
    lr = args.lr
    l2_coef = args.l2_coef
    hid_dim = args.hidden
    sample_size = args.sample_size
    num_hop = args.n
    test_conv_num = args.n

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("./logs/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('./logs/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # set up checkpoint path
    Path("./checkpoints/").mkdir(parents=True, exist_ok=True)

    logger.info('*******n is' + str(test_conv_num))
    accs = []
    for __ in range(n_runs):
        run_acc = train(dataset,
                        epochs,
                        patience,
                        lr,
                        l2_coef,
                        hid_dim,
                        sample_size,
                        num_hop,
                        test_conv_num,
                        args.cuda)
        accs.append(run_acc)

    with open('gslog_{}.txt'.format(args.data), 'a') as f:
        f.write(str(args))
        f.write('\n' + str(np.mean(accs)) + '\n')
        f.write(str(np.std(accs)) + '\n')