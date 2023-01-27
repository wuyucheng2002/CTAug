import torch
import networkx as nx
import numpy as np
import logging
from torch_geometric.utils import dropout_adj
from .pGRACE.functional import drop_edge_weighted


def norm_by_row(value):
    value = value - value.min()
    value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return value


def config_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_path, mode='w')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    return logger


def drop_edge(drop_weights, idx, data, param):

    if param['drop_scheme'] == 'uniform':
        return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
    elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
        return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'],
                                       threshold=0.7)
    else:
        raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')


class GetCore:
    def __init__(self, data, core, device, frac):
        self.data = data
        self.frac = frac
        self.device = device
        self.core = core
        self.weight = None

        if core in ['kcore', 'random']:
            self.core_num = self.get_main_order('kcore')
            self.cores0 = self.get_kcore(data.edge_index, 0)
            self.cores1 = self.get_kcore(data.edge_index, 1)
            self.cores2 = self.get_kcore(data.edge_index, 2)
        if core in ['ktruss', 'random']:
            self.truss_num = self.get_main_order('ktruss')
            self.trusses0 = self.get_ktruss(data.edge_index, 0)
            self.trusses1 = self.get_ktruss(data.edge_index, 1)
            self.trusses2 = self.get_ktruss(data.edge_index, 2)

    def get_main_order(self, core):
        edge_index = self.data.edge_index.T.cpu().detach().numpy()
        G = nx.from_edgelist(edge_index)
        core_order = max(nx.core_number(G).values())
        if core == 'kcore':
            return core_order
        else:
            if G.nodes:
                k = core_order + 1
                while k >= 2:
                    H = nx.k_truss(G, k)
                    if H.nodes:
                        break
                    k -= 1
                return k
            else:
                return 0

    def get_kcore(self, edge_index, d):
        edge_index = edge_index.T.cpu().detach().numpy()
        G = nx.from_edgelist(edge_index)
        if G.nodes:
            k = max(self.core_num - d, 0)
            H = nx.k_core(G, k)
            return list(H.nodes)
        else:
            return []

    def get_ktruss(self, edge_index, d):
        edge_index = edge_index.T.cpu().detach().numpy()
        G = nx.from_edgelist(edge_index)
        if G.nodes:
            k = max(self.truss_num  - d, 0)
            H = nx.k_truss(G, k)
            return list(H.nodes)
        else:
            return []

    def save_nodes(self, core, order):
        if order == 0:
            return self.cores0 if core == 'kcore' else self.trusses0
        elif order == 1:
            return self.cores1 if core == 'kcore' else self.trusses1
        else:
            return self.cores2 if core == 'kcore' else self.trusses2

    def get_prob(self, core, order):
        save = self.save_nodes(core, order)
        probs = torch.ones(self.data.edge_index.shape[1]).float().to(self.device)
        probs[save] = self.frac
        edge_probs = (probs[self.data.edge_index[0]] + probs[self.data.edge_index[1]]) / 2
        return edge_probs

    def preprocess(self):
        self.weight = []
        self.weight.append(self.get_prob(self.core, 0))
        self.weight.append(self.get_prob(self.core, 1))
        self.weight.append(self.get_prob(self.core, 2))


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