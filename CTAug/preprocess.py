import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import numpy as np
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv
from collections import Counter
import os


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def compute_ppr(a, alpha=0.2, self_loop=True):
    # a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def encode(graphs, id_encoding=True, degree_encoding=None):
    '''
        Encodes categorical variables such as structural identifiers and degree features.
    '''
    # encoder_ids, d_id = None, [1] * graphs[0].identifiers.shape[1]
    if id_encoding is not None:
        ids = [graph.identifiers for graph in graphs]
        encoder_ids = one_hot_unique(ids)
        d_id = encoder_ids.d
        encoder_ids = encoder_ids.fit(ids)

    # encoder_degrees, d_degree = None, []
    if degree_encoding is not None:
        degrees = [graph.degrees.unsqueeze(1) for graph in graphs]
        encoder_degrees = one_hot_unique(degrees)
        # d_degree = encoder_degrees.d
        encoded_degrees = encoder_degrees.fit(degrees)

    for g, graph in enumerate(graphs):
        if id_encoding is not None:
            setattr(graph, 'identifiers', encoder_ids[g])
        if degree_encoding is not None:
            setattr(graph, 'degrees', encoded_degrees[g])
    return graphs, d_id


class one_hot_unique:
    def __init__(self, tensor_list):
        tensor_list = torch.cat(tensor_list, 0)
        self.d = list()
        self.corrs = dict()
        for col in range(tensor_list.shape[1]):
            uniques, corrs = np.unique(tensor_list[:, col], return_inverse=True, axis=0)
            self.d.append(len(uniques))
            self.corrs[col] = corrs
        return

    def fit(self, tensor_list):
        pointer = 0
        encoded = None
        encoded_tensors = list()
        for tensor in tensor_list:
            n = tensor.shape[0]
            for col in range(tensor.shape[1]):
                translated = torch.LongTensor(self.corrs[col][pointer:pointer + n]).unsqueeze(1)
                encoded = torch.cat((encoded, translated), 1) if col > 0 else translated
            encoded_tensors.append(encoded)
            pointer += n
        return encoded_tensors


def get_dataset_sub(dataset_file):
    dataset = torch.load('data/' + dataset_file)[0]
    dataset, d_id = encode(dataset)
    for i, data in enumerate(dataset):
        dataset[i].idx = i
        subs = data.identifiers
        onehots = []
        for j in range(subs.shape[1]):
            onehot = torch.zeros((subs.shape[0], d_id[j]), device=subs.device)
            onehot.scatter_(1, subs[:, j:j+1], 1)
            onehots.append(onehot)
        dataset[i].x = torch.cat(onehots, 1)
    # print(dataset)
    return dataset, dataset[0].x.shape[1], dataset[0].x.shape[1]


def get_dataset_one(name):
    dataset = TUDataset('data', name)
    dataset_with_id = []
    for i in range(len(dataset)):
        data = dataset[i]
        data.idx = i
        if data.x is None:
            data.x = torch.ones((data.num_nodes, 1)).float()
        else:
            data.x = torch.ones((data.x.shape[0], 1)).float()
        dataset_with_id.append(data)
    return dataset_with_id, 1, 0


def get_dataset_deg(name):
    dataset = TUDataset('data', name)
    dataset_with_id = []
    maxd = torch.tensor(100)
    for i in range(len(dataset)):
        data = dataset[i]
        data.idx = i
        row, _ = data.edge_index
        if data.x is None:
            num = data.num_nodes
        else:
            num = data.x.shape[0]
        deg = degree(row, num).view((-1, 1))
        deg_capped = torch.min(deg, maxd).type(torch.int64)
        deg_onehot = F.one_hot(deg_capped.view(-1), num_classes=int(maxd.item()) + 1).type(deg.dtype)
        data.x = deg_onehot
        dataset_with_id.append(data)
    return dataset_with_id, dataset_with_id[0].x.shape[1], 0


def get_dataset_sub_deg(dataset_file):
    dataset = torch.load('data/' + dataset_file)[0]
    dataset, d_id = encode(dataset)
    maxd = torch.tensor(100)
    dataset_with_id = []
    for i in range(len(dataset)):
        data = dataset[i]
        data.idx = i
        subs = data.identifiers
        onehots = []
        for j in range(subs.shape[1]):
            onehot = torch.zeros((subs.shape[0], d_id[j]), device=subs.device)
            onehot.scatter_(1, subs[:, j:j + 1], 1)
            onehots.append(onehot)
        onehots = torch.cat(onehots, 1)

        # deg = data.degrees.view((-1, 1))
        row, _ = data.edge_index
        deg = degree(row, data.x.shape[0]).view((-1, 1))
        deg_capped = torch.min(deg, maxd).type(torch.int64)
        deg_onehot = F.one_hot(deg_capped.view(-1), num_classes=int(maxd.item()) + 1).type(deg.dtype)

        data.x = torch.cat((onehots, deg_onehot), dim=1)
        dataset_with_id.append(data)
    # print(dataset)
    return dataset, data.x.shape[1], onehots.shape[1]


def preprocess(dataset, dataset_file, cal_weight=None):
    id_list = []
    if dataset_file is None:
        dataset_id = TUDataset('data', dataset)
        dim = 0
    else:
        dataset_id = torch.load('data/' + dataset_file)[0]
        dataset_id, d_id = encode(dataset_id)
        for i, data in enumerate(dataset_id):
            subs = data.identifiers
            onehots = []
            for j in range(subs.shape[1]):
                onehot = torch.zeros((subs.shape[0], d_id[j]), device=subs.device)
                onehot.scatter_(1, subs[:, j:j + 1], 1)
                onehots.append(onehot)
            id_list.append(torch.cat(onehots, 1).cpu().detach().numpy())
        dim = id_list[0].shape[1]

    graphs = []
    labels, feats = [], []
    for idx, data in enumerate(dataset_id):
        edge_index = data.edge_index.T.cpu().detach().numpy()
        graph = nx.from_edgelist(edge_index)
        graph.graph['label'] = data.y.item()

        for u in graph.nodes(data=True):
            f = np.zeros(63 + 1)
            f[min(graph.degree[u[0]], 63)] = 1.0
            if id_list:
                f = np.concatenate((id_list[idx][u[0], :], f), axis=-1)
            graph.nodes[u[0]]['feat'] = f
        labels.append(graph.graph['label'])
        feats.append(np.array(list(nx.get_node_attributes(graph, 'feat').values())))

        # relabeling
        mapping = {}
        for node_idx, node in enumerate(graph.nodes()):
            mapping[node] = node_idx
        graphs.append(nx.relabel_nodes(graph, mapping))

    for graph in graphs:
        if cal_weight == 'node':
            for u in graph.nodes:
                graph.nodes[u]['kcore'] = 0
                graph.nodes[u]['ktruss'] = 0
            # kcore
            H = graph
            k = 1
            while H.nodes:
                H = nx.k_core(graph, k)
                for n in H.nodes:
                    graph.nodes[n]['kcore'] += 1
                k += 1
            # ktruss
            H = graph
            k = 1
            while H.nodes:
                H = nx.k_truss(graph, k)
                for n in H.nodes:
                    graph.nodes[n]['ktruss'] += 1
                k += 1
            # average node weight
            graph.graph['n_kcore'] = np.mean([graph.nodes[n]['kcore'] for n in graph.nodes])
            graph.graph['n_ktruss'] = np.mean([graph.nodes[n]['ktruss'] for n in graph.nodes])

        if cal_weight == 'edge':
            for e in graph.edges:
                graph.edges[e]['kcore'] = 0
                graph.edges[e]['ktruss'] = 0
            # kcore
            H = graph
            k = 1
            while H.edges:
                H = nx.k_core(graph, k)
                for e in H.edges:
                    graph.edges[e]['kcore'] += 1
                k += 1
            # ktruss
            H = graph
            k = 1
            while H.edges:
                H = nx.k_truss(graph, k)
                for e in H.edges:
                    graph.edges[e]['ktruss'] += 1
                k += 1
            # average edge weight
            graph.graph['e_kcore'] = np.mean([graph.edges[n]['kcore'] for n in graph.edges])
            graph.graph['e_ktruss'] = np.mean([graph.edges[n]['ktruss'] for n in graph.edges])

    return graphs, np.array(feats), np.array(labels), dim


def process_graph(graphs, device, kcore=None, ktruss=None, random=None, cal_weight=None):
    adjs, diffs = [], []
    for graph in graphs:
        if cal_weight == 'node':
            m_kcore = graph.graph['n_kcore']
            m_ktruss = graph.graph['n_ktruss']
            for n in graph.nodes:
                graph.nodes[n]['weight'] = graph.nodes[n]['kcore'] / m_kcore * kcore + \
                                           graph.nodes[n]['ktruss'] / m_ktruss * ktruss + random * 1
            for e in graph.edges:
                graph.edges[e]['weight'] = (graph.nodes[e[0]]['weight'] + graph.nodes[e[1]]['weight']) / 2

        if cal_weight == 'edge':
            m_kcore = graph.graph['e_kcore']
            m_ktruss = graph.graph['e_ktruss']
            for e in graph.edges:
                graph.edges[e]['weight'] = graph.edges[e]['kcore'] / m_kcore * kcore + \
                                           graph.edges[e]['ktruss'] / m_ktruss * ktruss + random * 1
        adj = nx.to_numpy_array(graph)
        adjs.append(adj)
        diffs.append(compute_ppr(adj, alpha=0.2))

    max_nodes = max([a.shape[0] for a in adjs])
    for idx in range(len(adjs)):
        adjs[idx] = normalize_adj(adjs[idx]).todense()

        diffs[idx] = np.hstack(
            (np.vstack((diffs[idx], np.zeros((max_nodes - diffs[idx].shape[0], diffs[idx].shape[0])))),
             np.zeros((max_nodes, max_nodes - diffs[idx].shape[1]))))

        adjs[idx] = np.hstack(
            (np.vstack((adjs[idx], np.zeros((max_nodes - adjs[idx].shape[0], adjs[idx].shape[0])))),
             np.zeros((max_nodes, max_nodes - adjs[idx].shape[1]))))

    adjs = torch.FloatTensor(np.array(adjs).reshape(-1, max_nodes, max_nodes)).to(device)
    diffs = torch.FloatTensor(np.array(diffs).reshape(-1, max_nodes, max_nodes)).to(device)

    return adjs, diffs, max_nodes
