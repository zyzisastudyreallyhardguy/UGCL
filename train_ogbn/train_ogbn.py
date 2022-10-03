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
from model import UGCL, GCN, LogReg
from dataset import load, load_ogb
from torch_geometric.seed import seed_everything
import dgl

def test_evaluator(model, adj, idx_train, idx_val, idx_test, features, labels, hid_units, nb_classes, xent, cuda_no, num_hop, test_epoch):
    features = torch.FloatTensor(features).cuda(cuda_no)
    adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj)).cuda(cuda_no)

    embeds = model.embed(features, adj, True, num_hop).cuda(cuda_no)

    embeds = normalize(embeds, p=2.0, dim=2) # noramlisation before test evaluator

    train_embs = embeds[0, idx_train]

    train_lbls = labels[idx_train]
    accs = []
    val_accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0

    log = LogReg(hid_units, nb_classes).cuda(cuda_no)
    opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)

    for _ in range(test_epoch):
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

    val_acc = torch.sum(preds[idx_val].argmax(1) == labels[idx_val]).float() / idx_val.shape[0]
    val_accs.append(val_acc * 100)
    val_accs = torch.tensor(val_accs)

    # logger.info('evaluation acc is {} with std {}'.format(accs.mean().item(), accs.std().item()))
    return accs.mean().item(), val_accs.mean().item()

def train(dataset, epochs, patience, lr, l2_coef, hid_dim, sample_size, num_hop, cuda_no, test_epoch, verbose=True):
    # parameter setting portal
    nb_epochs = epochs
    patience = patience
    batch_size = 1
    lr = lr
    l2_coef = l2_coef
    hid_units = hid_dim
    sample_size = sample_size
    sparse = True

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

    #Load dataset
    adj, features, labels, idx_train, idx_val, idx_test = load_ogb(args)

    #Initialization
    features = features.numpy()

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    model = UGCL(ft_size, hid_units)
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

    #training
    for epoch in range(nb_epochs):
        #subsampling
        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size].todense())
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba[0]))
        else:
            ba = torch.FloatTensor(ba)

        bf = torch.FloatTensor(bf)

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()

        model.train()
        optimiser.zero_grad()

        loss = model(bf, ba, num_hop, sparse)

        print('epoch:' + str(epoch) + '|loss:' + str(loss))

        loss.backward()
        optimiser.step()

        # logger.info('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if (epoch == (args.epoch - 1)) and verbose:
        # if verbose:
            model.eval()
            acc, val_acc = test_evaluator(model, adj, idx_train, idx_val, idx_test, features, labels, hid_units, nb_classes,
                                 xent, cuda_no, num_hop, test_epoch)
            print('test acc:' + str(acc))
            print('val acc:' + str(val_acc))
            if acc > best_acc:
                best_acc = acc

        #save model
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

    return best_acc

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser('UGCL')
    parser.add_argument('--data', type=str, default='ogbn', help='Dataset name: cora, citeseer, pubmed')
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
    parser.add_argument('--test_epoch', type=int, default=2000, help='epoches for testing')
    parser.add_argument('--data_root_dir', type=str, default='default',
                           help="dir_path for saving graph data. Note that this model use DGL loader so do not mix up with the dir_path for the Pyg one. Use 'default' to save datasets at current folder.")

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
    test_epoch = args.test_epoch

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
                        args.cuda,
                        test_epoch)
        accs.append(run_acc)
        # logger.info('------n:' + str(test_conv_num) + '-----')
        # logger.info('accs are: {}'.format(accs))
        # logger.info('Final average acc is {} with std {}'.format(np.mean(accs), np.std(accs)))
        # logger.info('test learning rate: ' + str(lr))

    with open('gslog_{}.txt'.format(args.data), 'a') as f:
        f.write(str(args))
        f.write('\n' + str(np.mean(accs)) + '\n')
        f.write(str(np.std(accs)) + '\n')