import networkx as nx
import GCL.augmentors as A
import torch
import numpy as np


def get_aug_p(data, aug_p, pn):
    num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
    n = np.random.choice(3, 1, p=aug_p)[0]
    if n == 0:
        return A.NodeDropping(pn=pn)
    elif n == 1:
        return A.EdgeRemoving(pe=pn)
    elif n == 2:
        return A.RWSampling(num_seeds=num_seeds, walk_length=10)
    else:
        NameError('Augmentation is wrong.')


def get_aug_p2(aug_p, pn):
    # num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
    n = np.random.choice(2, 1, p=aug_p)[0]
    if n == 0:
        return A.NodeDropping(pn=pn)
    elif n == 1:
        return A.EdgeRemoving(pe=pn)
    else:
        NameError('Augmentation is wrong.')


def drop_edge(data, pn):
    keep_prob = 1. - pn
    probs = torch.tensor([keep_prob for _ in range(data.edge_index.shape[1])]).to(data.x.device)
    edge_mask = torch.bernoulli(probs).to(torch.bool).to(data.x.device)
    edge_index1 = data.edge_index[:, edge_mask]
    return data.x, edge_index1, data.batch


def drop_node(data, pn):
    keep_prob = 1. - pn
    probs = torch.tensor([keep_prob for _ in range(data.x.shape[0])]).to(data.x.device)

    while True:
        node_mask = torch.bernoulli(probs).to(torch.bool).to(data.x.device)
        batch1 = data.batch[node_mask]
        if set(batch1.tolist()) == set(data.batch.tolist()):
            break

    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    edge_index1 = data.edge_index[:, edge_mask]

    # relabel
    x1 = data.x[node_mask]
    node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=data.x.device)
    node_idx[node_mask] = torch.arange(node_mask.sum().item(), device=data.x.device)
    edge_index1 = node_idx[edge_index1]
    return x1, edge_index1, batch1


class GetCore:
    def __init__(self, dataset, core, pn, frac, device):
        self.dataset = dataset
        self.pn = pn
        self.frac = frac
        self.device = device

        if core == 'kcore':
            self.core_num = self.get_main_order('kcore')
            self.cores0 = self.get_all('kcore', 0)
            self.cores1 = self.get_all('kcore', 1)
            self.cores2 = self.get_all('kcore', 2)
        if core == 'ktruss':
            self.truss_num = self.get_main_order('ktruss')
            self.trusses0 = self.get_all('ktruss', 0)
            self.trusses1 = self.get_all('ktruss', 1)
            self.trusses2 = self.get_all('ktruss', 2)

    def extract_core(self, edge_index, core, d, idx):
        if edge_index.shape[1] != 0:
            edge_index = edge_index.T.cpu().detach().numpy()
            G = nx.from_edgelist(edge_index)
            if core == 'kcore':
                k = max(self.core_num[idx] - d, 0)
                H = nx.k_core(G, k)
                return list(H.nodes)
            else:
                k = max(self.truss_num[idx] - d, 2)
                H = nx.k_truss(G, k)
                return list(H.nodes)
        else:
            return []

    def get_all(self, core, order):
        alls = []
        for idx, data in enumerate(self.dataset):
            alls.append(self.extract_core(data.edge_index, core, order, idx))
        return alls

    def get_main_order(self, core):
        orders = []
        for idx, data in enumerate(self.dataset):
            edge_index = data.edge_index.T.cpu().detach().numpy()
            G = nx.from_edgelist(edge_index)
            core_order = max(nx.core_number(G).values())
            if core == 'kcore':
                orders.append(core_order)
            else:
                if G.nodes:
                    k = core_order + 1
                    while k >= 2:
                        H = nx.k_truss(G, k)
                        if H.nodes:
                            break
                        k -= 1
                    orders.append(k)
                else:
                    orders.append(2)
        return orders

    def save_nodes(self, core, order, idx):
        if order == 0:
            return self.cores0[idx] if core == 'kcore' else self.trusses0[idx]
        elif order == 1:
            return self.cores1[idx] if core == 'kcore' else self.trusses1[idx]
        else:
            return self.cores2[idx] if core == 'kcore' else self.trusses2[idx]

    def drop_node(self, x, edge_index, num_nodes, core, order, idx):
        keep_prob = 1. - self.pn
        probs = torch.tensor([keep_prob for _ in range(num_nodes)]).to(self.device)
        save = self.save_nodes(core, order, idx)
        probs[save] = 1. - self.frac * self.pn

        while True:
            node_mask = torch.bernoulli(probs).to(torch.bool).to(self.device)
            num_node = node_mask.sum().item()
            if num_node > 0:
                break

        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index1 = edge_index[:, edge_mask]

        # relabel
        x1 = x[node_mask]
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=self.device)
        node_idx[node_mask] = torch.arange(num_node, device=self.device)
        edge_index1 = node_idx[edge_index1]
        return x1, edge_index1, num_node

    def drop_edge(self, edge_index, num_nodes, core, order, idx):
        keep_prob = 1. - self.pn
        probs = torch.tensor([keep_prob for _ in range(num_nodes)])
        save = self.save_nodes(core, order, idx)
        probs[save] = 1. - self.frac * self.pn
        edge_probs = (probs[edge_index[0]] + probs[edge_index[1]]) / 2
        edge_mask = torch.bernoulli(edge_probs).to(torch.bool)
        edge_index1 = edge_index[:, edge_mask]
        return edge_index1

    def get_graph(self, data, core, order):
        xs = []
        edge_indexs = []
        batchs = []
        nodes = 0
        for k in range(data.y.size(0)):
            num_nodes = data.ptr[k + 1] - data.ptr[k]
            edge_index = data.edge_index[:, (data.edge_index[0, :] >= data.ptr[k]) &
                                            (data.edge_index[0, :] < data.ptr[k + 1])] - data.ptr[k]
            # x = data.x[data.ptr[k]: data.ptr[k + 1], :]
            x = data.x[data.batch == k]
            x1, edge_index1, num_node = self.drop_node(x, edge_index, num_nodes, core, order, data.idx[k])
            xs.append(x1)
            edge_indexs.append(edge_index1 + nodes)
            batchs.append(torch.ones(num_node, dtype=torch.long, device=self.device) * k)
            nodes += num_node
        xs = torch.cat(xs, dim=0).to(self.device)
        edge_indexs = torch.cat(edge_indexs, dim=-1).long().to(self.device)
        batchs = torch.cat(batchs, dim=-1).long().to(self.device)
        return xs, edge_indexs, batchs

    # def get_graph(self, data, core, order):
    #     edge_indexs = []
    #     for k in range(data.y.size(0)):
    #         num_nodes = data.ptr[k + 1] - data.ptr[k]
    #         edge_index = data.edge_index[:, (data.edge_index[0, :] >= data.ptr[k]) &
    #                                         (data.edge_index[0, :] < data.ptr[k + 1])] - data.ptr[k]
    #         edge_index1 = self.drop_edge(edge_index, num_nodes, core, order, data.idx[k])
    #         edge_indexs.append(edge_index1 + data.ptr[k])
    #     edge_indexs = torch.cat(edge_indexs, dim=-1).long().to(self.device)
    #     return data.x, edge_indexs, data.batch


class GetCoreJ:
    def __init__(self, dataset, core, pn, frac, device):
        self.dataset = dataset
        self.pn = pn
        self.frac = frac
        self.device = device

        if core == 'kcore':
            self.core_num = self.get_main_order('kcore')
            self.cores0 = self.get_all('kcore', 0)
            self.cores1 = self.get_all('kcore', 1)
            self.cores2 = self.get_all('kcore', 2)
        if core == 'ktruss':
            self.truss_num = self.get_main_order('ktruss')
            self.trusses0 = self.get_all('ktruss', 0)
            self.trusses1 = self.get_all('ktruss', 1)
            self.trusses2 = self.get_all('ktruss', 2)

    def extract_core(self, edge_index, core, d, idx):
        if edge_index.shape[1] != 0:
            edge_index = edge_index.T.cpu().detach().numpy()
            G = nx.from_edgelist(edge_index)
            if core == 'kcore':
                k = max(self.core_num[idx] - d, 0)
                H = nx.k_core(G, k)
                return list(H.nodes)
            else:
                k = max(self.truss_num[idx] - d, 2)
                H = nx.k_truss(G, k)
                return list(H.nodes)
        else:
            return []

    def get_all(self, core, order):
        alls = []
        for idx, data in enumerate(self.dataset):
            alls.append(self.extract_core(data.edge_index, core, order, idx))
        return alls

    def get_main_order(self, core):
        orders = []
        for idx, data in enumerate(self.dataset):
            edge_index = data.edge_index.T.cpu().detach().numpy()
            G = nx.from_edgelist(edge_index)
            core_order = max(nx.core_number(G).values())
            if core == 'kcore':
                orders.append(core_order)
            else:
                if G.nodes:
                    k = core_order + 1
                    while k >= 2:
                        H = nx.k_truss(G, k)
                        if H.nodes:
                            break
                        k -= 1
                    orders.append(k)
                else:
                    orders.append(2)
        return orders

    def save_nodes(self, core, order, idx):
        if order == 0:
            return self.cores0[idx] if core == 'kcore' else self.trusses0[idx]
        elif order == 1:
            return self.cores1[idx] if core == 'kcore' else self.trusses1[idx]
        else:
            return self.cores2[idx] if core == 'kcore' else self.trusses2[idx]

    def drop_node(self, x, edge_index, num_nodes, core, order, idx):
        keep_prob = 1. - self.pn
        probs = torch.tensor([keep_prob for _ in range(num_nodes)]).to(self.device)
        save = self.save_nodes(core, order, idx)
        probs[save] = 1. - self.frac * self.pn

        while True:
            node_mask = torch.bernoulli(probs).to(torch.bool).to(self.device)
            num_node = node_mask.sum().item()
            if num_node > 0:
                break

        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index1 = edge_index[:, edge_mask]

        # relabel
        x1 = x[node_mask]
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=self.device)
        node_idx[node_mask] = torch.arange(num_node, device=self.device)
        edge_index1 = node_idx[edge_index1]
        return x1, edge_index1, num_node

    def drop_edge(self, edge_index, num_nodes, core, order, idx):
        keep_prob = 1. - self.pn
        probs = torch.tensor([keep_prob for _ in range(num_nodes)]).to(self.device)
        save = self.save_nodes(core, order, idx)
        probs[save] = 1. - self.frac * self.pn
        edge_probs = (probs[edge_index[0]] + probs[edge_index[1]]) / 2
        edge_mask = torch.bernoulli(edge_probs).to(torch.bool)
        edge_index1 = edge_index[:, edge_mask]
        return edge_index1

    def get_graph_node(self, data, core, order):
        xs = []
        edge_indexs = []
        batchs = []
        nodes = 0
        for k in range(data.y.size(0)):
            num_nodes = data.ptr[k + 1] - data.ptr[k]
            edge_index = data.edge_index[:, (data.edge_index[0, :] >= data.ptr[k]) &
                                            (data.edge_index[0, :] < data.ptr[k + 1])] - data.ptr[k]
            # x = data.x[data.ptr[k]: data.ptr[k + 1], :]
            x = data.x[data.batch == k]

            x1, edge_index1, num_node = self.drop_node(x, edge_index, num_nodes, core, order, data.idx[k])
            xs.append(x1)
            edge_indexs.append(edge_index1 + nodes)
            batchs.append(torch.ones(num_node, dtype=torch.long, device=self.device) * k)
            nodes += num_node
        xs = torch.cat(xs, dim=0).to(self.device)
        edge_indexs = torch.cat(edge_indexs, dim=-1).long().to(self.device)
        batchs = torch.cat(batchs, dim=-1).long().to(self.device)
        return xs, edge_indexs, batchs

    def get_graph_edge(self, data, core, order):
        edge_indexs = []
        for k in range(data.y.size(0)):
            num_nodes = data.ptr[k + 1] - data.ptr[k]
            edge_index = data.edge_index[:, (data.edge_index[0, :] >= data.ptr[k]) &
                                            (data.edge_index[0, :] < data.ptr[k + 1])] - data.ptr[k]
            edge_index1 = self.drop_edge(edge_index, num_nodes, core, order, data.idx[k])
            edge_indexs.append(edge_index1 + data.ptr[k])
        edge_indexs = torch.cat(edge_indexs, dim=-1).long().to(self.device)
        return data.x, edge_indexs, data.batch

    def get_graph(self, data, core, order, aug_p):
        n = np.random.choice(2, 1, p=aug_p)[0]
        if n == 0:
            return self.get_graph_node(data, core, order)
        if n == 1:
            return self.get_graph_edge(data, core, order)
        else:
            NameError('Augmentation is wrong.')


# class GetCoreJ3:
#     def __init__(self, dataset, core, pn, frac, device):
#         self.dataset = dataset
#         self.pn = pn
#         self.frac = frac
#         self.device = device
#
#         if core in ['kcore', 'random']:
#             self.core_num = self.get_main_order('kcore')
#             self.cores0 = self.get_all('kcore', 0)
#             self.cores1 = self.get_all('kcore', 1)
#             self.cores2 = self.get_all('kcore', 2)
#         if core in ['ktruss', 'random']:
#             self.truss_num = self.get_main_order('ktruss')
#             self.trusses0 = self.get_all('ktruss', 0)
#             self.trusses1 = self.get_all('ktruss', 1)
#             self.trusses2 = self.get_all('ktruss', 2)
#
#     def extract_core(self, edge_index, core, d, idx):
#         if edge_index.shape[1] != 0:
#             edge_index = edge_index.T.cpu().detach().numpy()
#             G = nx.from_edgelist(edge_index)
#             if core == 'kcore':
#                 k = max(self.core_num[idx] - d, 0)
#                 H = nx.k_core(G, k)
#                 return list(H.nodes)
#             else:
#                 k = max(self.truss_num[idx] - d, 2)
#                 H = nx.k_truss(G, k)
#                 return list(H.nodes)
#         else:
#             return []
#
#     def get_all(self, core, order):
#         alls = []
#         for idx, data in enumerate(self.dataset):
#             alls.append(self.extract_core(data.edge_index, core, order, idx))
#         return alls
#
#     def get_main_order(self, core):
#         orders = []
#         for idx, data in enumerate(self.dataset):
#             edge_index = data.edge_index.T.cpu().detach().numpy()
#             G = nx.from_edgelist(edge_index)
#             core_order = max(nx.core_number(G).values())
#             if core == 'kcore':
#                 orders.append(core_order)
#             else:
#                 if G.nodes:
#                     k = core_order + 1
#                     while k >= 2:
#                         H = nx.k_truss(G, k)
#                         if H.nodes:
#                             break
#                         k -= 1
#                     orders.append(k)
#                 else:
#                     orders.append(2)
#         return orders
#
#     def save_nodes(self, core, order, idx):
#         if order == 0:
#             return self.cores0[idx] if core == 'kcore' else self.trusses0[idx]
#         elif order == 1:
#             return self.cores1[idx] if core == 'kcore' else self.trusses1[idx]
#         else:
#             return self.cores2[idx] if core == 'kcore' else self.trusses2[idx]
#
#     def drop_node(self, x, edge_index, num_nodes, core, order, idx):
#         keep_prob = 1. - self.pn
#         probs = torch.tensor([keep_prob for _ in range(num_nodes)])
#         save = self.save_nodes(core, order, idx)
#         probs[save] = 1. - self.frac * self.pn
#
#         while True:
#             node_mask = torch.bernoulli(probs).to(torch.bool)
#             num_node = node_mask.sum().item()
#             if num_node > 0:
#                 break
#
#         edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
#         edge_index1 = edge_index[:, edge_mask]
#
#         # relabel
#         x1 = x[node_mask]
#         node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=self.device)
#         node_idx[node_mask] = torch.arange(num_node, device=self.device)
#         edge_index1 = node_idx[edge_index1]
#         return x1, edge_index1, num_node
#
#     def drop_edge(self, edge_index, num_nodes, core, order, idx):
#         keep_prob = 1. - self.pn
#         probs = torch.tensor([keep_prob for _ in range(num_nodes)])
#         save = self.save_nodes(core, order, idx)
#         probs[save] = 1. - self.frac * self.pn
#         edge_probs = (probs[edge_index[0]] + probs[edge_index[1]]) / 2
#         edge_mask = torch.bernoulli(edge_probs).to(torch.bool)
#         edge_index1 = edge_index[:, edge_mask]
#         return edge_index1
#
#     def get_graph_node(self, data, core, order):
#         xs = []
#         edge_indexs = []
#         batchs = []
#         nodes = 0
#         for k in range(data.y.size(0)):
#             num_nodes = data.ptr[k + 1] - data.ptr[k]
#             edge_index = data.edge_index[:, (data.edge_index[0, :] >= data.ptr[k]) &
#                                             (data.edge_index[0, :] < data.ptr[k + 1])] - data.ptr[k]
#             # x = data.x[data.ptr[k]: data.ptr[k + 1], :]
#             x = data.x[data.batch == k]
#
#             x1, edge_index1, num_node = self.drop_node(x, edge_index, num_nodes, core, order, data.idx[k])
#             xs.append(x1)
#             edge_indexs.append(edge_index1 + nodes)
#             batchs.append(torch.ones(num_node, dtype=torch.long, device=self.device) * k)
#             nodes += num_node
#         xs = torch.cat(xs, dim=0).to(self.device)
#         edge_indexs = torch.cat(edge_indexs, dim=-1).long().to(self.device)
#         batchs = torch.cat(batchs, dim=-1).long().to(self.device)
#         return xs, edge_indexs, batchs
#
#     def get_graph_edge(self, data, core, order):
#         edge_indexs = []
#         for k in range(data.y.size(0)):
#             num_nodes = data.ptr[k + 1] - data.ptr[k]
#             edge_index = data.edge_index[:, (data.edge_index[0, :] >= data.ptr[k]) &
#                                             (data.edge_index[0, :] < data.ptr[k + 1])] - data.ptr[k]
#             edge_index1 = self.drop_edge(edge_index, num_nodes, core, order, data.idx[k])
#             edge_indexs.append(edge_index1 + data.ptr[k])
#         edge_indexs = torch.cat(edge_indexs, dim=-1).long().to(self.device)
#         return data.x, edge_indexs, data.batch
#
#     def get_graph(self, data, core, order, aug_p):
#         n = np.random.choice(3, 1, p=aug_p)[0]
#         if n == 0:
#             return self.get_graph_node(data, core, order)
#         elif n == 1:
#             return self.get_graph_edge(data, core, order)
#         elif n == 2:
#             num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - self.pn), 0))
#             aug = A.RWSampling(num_seeds=num_seeds, walk_length=10)
#             _, edge_indexs, _ = aug(data.x, data.edge_index.to(torch.device('cpu')))
#             return data.x, edge_indexs.to(self.device), data.batch
#         else:
#             NameError('Augmentation is wrong.')


# def get_aug(data, aug_name, pn=0.2):
#     if data.edge_index.shape[1] == 0:
#         return A.Identity()
#     num_seeds = int(round(data.edge_index.shape[1] / 20 * (1 - pn), 0))
#     if aug_name == 'ND':
#         return A.NodeDropping(pn=pn)
#     elif aug_name == 'ER':
#         return A.EdgeRemoving(pe=pn)
#     elif aug_name == 'SUB':
#         return A.RWSampling(num_seeds=num_seeds, walk_length=10)
#     elif aug_name in ['ND+ER', 'ER+ND']:
#         x = random.choice([1, 2])
#         aug = A.NodeDropping(pn=pn) if x == 1 else A.EdgeRemoving(pe=pn)
#         return aug
#     elif aug_name in ['ND+SUB', 'SUB+ND']:
#         x = random.choice([1, 2])
#         aug = A.NodeDropping(pn=pn) if x == 1 else A.RWSampling(num_seeds=num_seeds, walk_length=10)
#         return aug
#     elif aug_name in ['ER+SUB', 'SUB+ER']:
#         x = random.choice([1, 2])
#         aug = A.EdgeRemoving(pe=pn) if x == 1 else A.RWSampling(num_seeds=num_seeds, walk_length=10)
#         return aug
#     elif aug_name in ['ND+ER+SUB', 'ND+SUB+ER', 'ER+ND+SUB', 'ER+SUB+ND', 'SUB+ND+ER', 'SUB+ER+ND']:
#         x = random.choice([1, 2, 3])
#         aug = A.NodeDropping(pn=pn) if x == 1 else \
#             A.EdgeRemoving(pe=pn) if x == 2 else A.RWSampling(num_seeds=num_seeds, walk_length=10)
#         return aug
#     else:
#         raise NameError('Augmentation is wrong.')
#
#
# def subgraph(edge_index, subset):
#     if len(subset) == 0 or len(edge_index.flatten()) == 0:
#         return []
#     else:
#         num_nodes = max(max(edge_index.flatten()), max(subset)) + 1
#         node_mask = np.zeros(num_nodes, dtype=bool)
#         node_mask[list(subset)] = 1
#         edge_mask = node_mask[edge_index[:, 0]] & node_mask[edge_index[:, 1]]
#         edge_index = edge_index[edge_mask, :]
#         return edge_index