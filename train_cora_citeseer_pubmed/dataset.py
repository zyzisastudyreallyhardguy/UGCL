import dgl
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
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
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
        to_undir = T.ToUndirected()
        data = to_undir(Planetoid(path, dataset, transform=T.NormalizeFeatures())[0])
        adj = to_dense_adj(data.edge_index)[0]
        idx_train = np.array(data.train_mask.nonzero().T.squeeze(0))
        idx_val = np.array(data.val_mask.nonzero().T.squeeze(0))
        idx_test = np.array(data.test_mask.nonzero().T.squeeze(0))
        feat = np.array(data.x)
        labels = np.array(data.y)
        return np.array(adj), feat, labels, idx_train, idx_val, idx_test
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

def load(dataset):
    datadir = os.path.join('data', dataset)

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        if dataset == 'cora':
            adj, feat, labels, idx_train, idx_val, idx_test = download(dataset)
        else:
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