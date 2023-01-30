# This code is mainly obtained from https://github.com/gbouritsas/GSN
import argparse
import types
from torch_geometric.datasets import TUDataset
import numpy as np
import networkx as nx
import sys
import graph_tool as gt
import graph_tool.topology as gt_topology
import torch
from torch_geometric.utils import remove_self_loops, degree, to_undirected
from torch_geometric.data import Data
import time
import os
import logging


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


def subgraph_counts2ids(count_fn, data, subgraph_dicts, subgraph_params):
    #### Remove self loops and then assign the structural identifiers by computing subgraph isomorphisms ####

    if hasattr(data, 'edge_features'):
        edge_index, edge_features = remove_self_loops(data.edge_index, data.edge_features)
        setattr(data, 'edge_features', edge_features)
    else:
        edge_index = remove_self_loops(data.edge_index)[0]

    if data.x is None:
        num_nodes = data.edge_index.max().item() + 1
    else:
        num_nodes = data.x.shape[0]
    identifiers = None
    logger.info("num of subgraph_dicts: {}".format(len(subgraph_dicts)))
    for i, subgraph_dict in enumerate(subgraph_dicts):
        logger.info(i)
        kwargs = {'subgraph_dict': subgraph_dict,
                  'induced': subgraph_params['induced'],
                  'num_nodes': num_nodes,
                  'directed': subgraph_params['directed']}
        counts = count_fn(edge_index, **kwargs)
        identifiers = counts if identifiers is None else torch.cat((identifiers, counts), 1)
    setattr(data, 'edge_index', edge_index)
    setattr(data, 'identifiers', identifiers.long())

    return data


def automorphism_orbits(edge_list, print_msgs=True, **kwargs):
    ##### vertex automorphism orbits #####

    directed = kwargs['directed'] if 'directed' in kwargs else False

    graph = gt.Graph(directed=directed)
    graph.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph)
    gt.stats.remove_parallel_edges(graph)

    # compute the vertex automorphism group
    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v

    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role

    orbit_membership_list = [[], []]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse=True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i, vertex in enumerate(orbit_membership_list[0])}

    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit] + [vertex]

    aut_count = len(aut_group)

    if print_msgs:
        logger.info('Orbit partition of given substructure: {}'.format(orbit_partition))
        logger.info('Number of orbits: {}'.format(len(orbit_partition)))
        logger.info('Automorphism count: {}'.format(aut_count))

    return graph, orbit_partition, orbit_membership, aut_count


def induced_edge_automorphism_orbits(edge_list, **kwargs):
    ##### induced edge automorphism orbits (according to the vertex automorphism group) #####

    directed = kwargs['directed'] if 'directed' in kwargs else False
    directed_orbits = kwargs['directed_orbits'] if 'directed_orbits' in kwargs else False

    graph, orbit_partition, orbit_membership, aut_count = automorphism_orbits(edge_list=edge_list,
                                                                              directed=directed,
                                                                              print_msgs=False)
    edge_orbit_partition = dict()
    edge_orbit_membership = dict()
    edge_orbits2inds = dict()
    ind = 0

    if not directed:
        edge_list = to_undirected(torch.tensor(graph.get_edges()).transpose(1, 0)).transpose(1, 0).tolist()

    # infer edge automorphisms from the vertex automorphisms
    for i, edge in enumerate(edge_list):
        if directed_orbits:
            edge_orbit = (orbit_membership[edge[0]], orbit_membership[edge[1]])
        else:
            edge_orbit = frozenset([orbit_membership[edge[0]], orbit_membership[edge[1]]])
        if edge_orbit not in edge_orbits2inds:
            edge_orbits2inds[edge_orbit] = ind
            ind_edge_orbit = ind
            ind += 1
        else:
            ind_edge_orbit = edge_orbits2inds[edge_orbit]

        if ind_edge_orbit not in edge_orbit_partition:
            edge_orbit_partition[ind_edge_orbit] = [tuple(edge)]
        else:
            edge_orbit_partition[ind_edge_orbit] += [tuple(edge)]

        edge_orbit_membership[i] = ind_edge_orbit

    print('Edge orbit partition of given substructure: {}'.format(edge_orbit_partition))
    print('Number of edge orbits: {}'.format(len(edge_orbit_partition)))
    print('Graph (vertex) automorphism count: {}'.format(aut_count))

    return graph, edge_orbit_partition, edge_orbit_membership, aut_count


def subgraph_isomorphism_vertex_counts(edge_index, **kwargs):
    ##### vertex structural identifiers #####

    subgraph_dict, induced, num_nodes = kwargs['subgraph_dict'], kwargs['induced'], kwargs['num_nodes']
    directed = kwargs['directed'] if 'directed' in kwargs else False

    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index.transpose(1, 0).cpu().numpy()))
    gt.stats.remove_self_loops(G_gt)
    gt.stats.remove_parallel_edges(G_gt)

    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=induced, subgraph=True,
                                               generator=True)

    ## num_nodes should be explicitly set for the following edge case:
    ## when there is an isolated vertex whose index is larger
    ## than the maximum available index in the edge_index

    counts = np.zeros((num_nodes, len(subgraph_dict['orbit_partition'])))
    for sub_iso_curr in sub_iso:
        for i, node in enumerate(sub_iso_curr):
            # increase the count for each orbit
            counts[node, subgraph_dict['orbit_membership'][i]] += 1
    counts = counts / subgraph_dict['aut_count']

    counts = torch.tensor(counts)

    return counts


def subgraph_isomorphism_edge_counts(edge_index, **kwargs):
    ##### edge structural identifiers #####

    subgraph_dict, induced = kwargs['subgraph_dict'], kwargs['induced']
    directed = kwargs['directed'] if 'directed' in kwargs else False

    edge_index = edge_index.transpose(1, 0).cpu().numpy()
    edge_dict = {}
    for i, edge in enumerate(edge_index):
        edge_dict[tuple(edge)] = i

    if not directed:
        subgraph_edges = to_undirected(
            torch.tensor(subgraph_dict['subgraph'].get_edges().tolist()).transpose(1, 0)).transpose(1, 0).tolist()

    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index))
    gt.stats.remove_self_loops(G_gt)
    gt.stats.remove_parallel_edges(G_gt)

    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=induced, subgraph=True,
                                               generator=True)

    counts = np.zeros((edge_index.shape[0], len(subgraph_dict['orbit_partition'])))

    for sub_iso_curr in sub_iso:
        mapping = sub_iso_curr.get_array()
        #         import pdb;pdb.set_trace()
        for i, edge in enumerate(subgraph_edges):
            # for every edge in the graph H, find the edge in the subgraph G_S to which it is mapped
            # (by finding where its endpoints are matched).
            # Then, increase the count of the matched edge w.r.t. the corresponding orbit
            # Repeat for the reverse edge (the one with the opposite direction)

            edge_orbit = subgraph_dict['orbit_membership'][i]
            mapped_edge = tuple([mapping[edge[0]], mapping[edge[1]]])
            counts[edge_dict[mapped_edge], edge_orbit] += 1

    counts = counts / subgraph_dict['aut_count']

    counts = torch.tensor(counts)

    return counts


def get_custom_edge_list(ks, substructure_type=None, filename=None):
    '''
        Instantiates a list of `edge_list`s representing substructures
        of type `substructure_type` with sizes specified by `ks`.
    '''
    if substructure_type is None and filename is None:
        raise ValueError('You must specify either a type or a filename where to read substructures from.')
    edge_lists = []
    for k in ks:
        if substructure_type is not None:
            graphs_nx = getattr(nx, substructure_type)(k)
        else:
            graphs_nx = nx.read_graph6(os.path.join(filename, 'graph{}c.g6'.format(k)))
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))
    return edge_lists


def main():
    assert args.id_type in ['cycle_graph', 'path_graph', 'complete_graph',
                            'binomial_tree', 'star_graph', 'nonisomorphic_trees']
    k_max = args.k
    k_min = 2 if args.id_type == 'star_graph' else 3
    custom_edge_list = get_custom_edge_list(list(range(k_min, k_max + 1)), args.id_type)

    automorphism_fn = induced_edge_automorphism_orbits if args.id_scope == 'local' else automorphism_orbits
    count_fn = subgraph_isomorphism_edge_counts if args.id_scope == 'local' else subgraph_isomorphism_vertex_counts

    subgraph_params = {'induced': False,
                       'edge_list': custom_edge_list,
                       'directed': False,
                       'directed_orbits': False}

    ### compute the orbits of earch substructure in the list, as well as the vertex automorphism count
    subgraph_dicts = []
    orbit_partition_sizes = []
    for edge_list in subgraph_params['edge_list']:
        subgraph, orbit_partition, orbit_membership, aut_count = \
            automorphism_fn(edge_list=edge_list, directed=subgraph_params['directed'],
                            directed_orbits=subgraph_params['directed_orbits'])
        subgraph_dicts.append({'subgraph': subgraph, 'orbit_partition': orbit_partition,
                               'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))

    if args.level == 'graph':
        dataset = TUDataset('data', args.dataset)

        graphs_ptg = []
        for i, data in enumerate(dataset):
            ii = i + 1
            logger.info("graph index: {}".format(ii))
            new_data = data
            if new_data.edge_index.shape[1] == 0:
                setattr(new_data, 'degrees', torch.zeros((new_data.graph_size,)))
            else:
                setattr(new_data, 'degrees', degree(new_data.edge_index[0]))
            new_data = subgraph_counts2ids(count_fn, new_data, subgraph_dicts, subgraph_params)
            graphs_ptg.append(new_data)

            if ii % 200 == 0:
                torch.save((graphs_ptg, orbit_partition_sizes), 'data/' + path + '_' + str(ii) + '.pt')
        torch.save((graphs_ptg, orbit_partition_sizes), 'data/' + path + '.pt')

    else:
        from torch_geometric.datasets import Coauthor, Amazon
        import torch_geometric.transforms as T
        if args.dataset == 'Coauthor-CS':
            dataset = Coauthor(root='data/' + args.dataset, name='cs', transform=T.NormalizeFeatures())
        elif args.dataset == 'Coauthor-Phy':
            dataset = Coauthor(root='data/' + args.dataset, name='physics', transform=T.NormalizeFeatures())
        elif args.dataset == 'Amazon-Computers':
            dataset = Amazon(root='data/' + args.dataset, name='computers', transform=T.NormalizeFeatures())
        else:
            raise NotImplementedError

        new_data = dataset[0]
        if new_data.edge_index.shape[1] == 0:
            setattr(new_data, 'degrees', torch.zeros((new_data.graph_size,)))
        else:
            setattr(new_data, 'degrees', degree(new_data.edge_index[0]))
        new_data = subgraph_counts2ids(subgraph_isomorphism_vertex_counts, new_data, subgraph_dicts,
                                       subgraph_params)
        torch.save((new_data, orbit_partition_sizes), 'data/' + path + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='IMDB-MULTI',
                        help="Dataset name, can be chosen from "
                             "graph classification: {'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', "
                             "                       'ENZYMES', 'PROTEINS'}, "
                             "node classification: {'Coauthor-Phy', 'Amazon-Computers'}.")
    parser.add_argument("--k", type=int, default=5,
                        help="count all the subgraphs of the family that have size up to k")
    parser.add_argument('--id_scope', type=str, default='global',
                        help="'local' vs 'global' --> GSN-e vs GSN-v (in O-GSN the default value is 'global')")
    parser.add_argument('--id_type', type=str, default='complete_graph',
                        help="Subgraphs family, can be chosen from"
                             "{'cycle_graph', 'path_graph', 'complete_graph', "
                             " 'binomial_tree', 'star_graph', 'nonisomorphic_trees'}")
    parser.add_argument('--level', type=str, default='graph',
                        help="The dataset is used for 'graph' classification or 'node' classification.")
    args = parser.parse_args()
    path = args.dataset + '_' + args.id_scope + '_' + args.id_type + '_' + str(args.k)
    logger = config_logger('log/' + path + '.log')
    main()
