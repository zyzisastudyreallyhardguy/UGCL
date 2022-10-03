import dgl
from dgl.data import CoraGraphDataset, CitationGraphDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from torch_geometric.transforms import RandomNodeSplit
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import torch
from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr, gdc
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os

def download(dataset):
    if dataset == 'cora':
        return CoraGraphDataset()
    elif dataset == 'citeseer':
        return CitationGraphDataset(name=dataset)
    elif dataset == 'pubmed':
        return CitationGraphDataset(name=dataset)
    elif dataset == 'computer':
        return AmazonCoBuyComputerDataset()
    elif dataset == 'photo':
        return AmazonCoBuyPhotoDataset()
    elif dataset == 'cs':
        return CoauthorCSDataset()
    elif dataset == 'phy':
        return CoauthorPhysicsDataset()
    else:
        return None

def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph

def load_data_ogb(dataset, args):
    global n_node_feats, n_classes

    if args.data_root_dir == 'default':
        data = DglNodePropPredDataset(name=dataset)
    else:
        data = DglNodePropPredDataset(name=dataset, root=args.data_root_dir)

    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator

def split_dataset(num_nodes, num_val, num_test):
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    if isinstance(num_val, float):
        num_val = round(num_nodes * num_val)
    else:
        num_val = num_val

    if isinstance(num_test, float):
        num_test = round(num_nodes * num_test)
    else:
        num_test = num_test

    perm = torch.randperm(num_nodes)
    val_mask[perm[:num_val]] = True
    test_mask[perm[num_val:num_val + num_test]] = True
    train_mask[perm[num_val + num_test:]] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    test_mask[remaining[num_val:num_val + num_test]] = True

    return train_mask, val_mask, test_mask


def load(dataset):
    datadir = os.path.join('data', dataset)

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(dataset)
        adj = nx.to_numpy_array(ds.graph)
        feat = ds.features[:]
        labels = ds.labels[:]

        idx_train = np.argwhere(ds.train_mask == 1).reshape(-1)
        idx_val = np.argwhere(ds.val_mask == 1).reshape(-1)
        idx_test = np.argwhere(ds.test_mask == 1).reshape(-1)

        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
        np.save(f'{datadir}/idx_train.npy', idx_train)
        np.save(f'{datadir}/idx_val.npy', idx_val)
        np.save(f'{datadir}/idx_test.npy', idx_test)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')
        idx_train = np.load(f'{datadir}/idx_train.npy')
        idx_val = np.load(f'{datadir}/idx_val.npy')
        idx_test = np.load(f'{datadir}/idx_test.npy')

    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        # epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        # avg_degree = np.sum(adj) / adj.shape[0]
        # epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
        #                               for e in epsilons])]
        #
        # diff[diff < epsilon] = 0.0
        # scaler = MinMaxScaler()
        # scaler.fit(diff)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, feat, labels, idx_train, idx_val, idx_test

def load_amco(dataset):
    datadir = os.path.join('data', dataset)

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(dataset)[0]
        row, col = ds.adj_sparse('coo')
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row.numpy(), col.numpy())), shape=(ds.num_nodes(), ds.num_nodes())).todense()
        # diff = compute_ppr(ds.graph, 0.2)
        feat = ds.ndata['feat'].numpy()
        labels = ds.ndata['label'].numpy()

        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')

    train_mask, val_mask, test_mask = split_dataset(adj.shape[0], 0.1, 0.8)
    idx_train = np.argwhere(train_mask == 1).reshape(-1)
    idx_val = np.argwhere(val_mask == 1).reshape(-1)
    idx_test = np.argwhere(test_mask == 1).reshape(-1)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, feat, labels, idx_train, idx_val, idx_test

def load_ogb(args):

    g, labels, idx_train, idx_val, idx_test, evaluator = load_data_ogb('ogbn-arxiv', args)
    g = preprocess(g)

    labels = labels.T.squeeze(0)

    feat = g.ndata['feat']

    row, col = g.adj_sparse('coo')
    adj = sp.coo_matrix((np.ones(row.shape[0]), (row.numpy(), col.numpy())),
                        shape=(g.num_nodes(), g.num_nodes()))

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).tocsr()

    return adj, feat, labels, idx_train, idx_val, idx_test
