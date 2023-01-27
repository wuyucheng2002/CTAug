import pickle
import os
import random
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected
from .utils_sp import SimpleParam
from .pGRACE.model import Encoder, GRACE, Encoder_OGSN
from .pGRACE.functional import drop_feature, degree_drop_weights, evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from .pGRACE.eval import log_regression, MulticlassEvaluator
from .pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from .pGRACE.dataset import get_dataset, get_dataset_no_norm
import numpy as np
import time
from .utils_ctaug import *


def train(model, optimizer, drop_weights, args, data, param, feature_weights):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = drop_edge(drop_weights, 1, data, param)
    edge_index_2 = drop_edge(drop_weights, 2, data, param)

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])
    else:
        x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(path_epo, model, data, split):
    model.eval()
    z = model(data.x, data.edge_index)

    x = z.cpu().detach().numpy()
    y = data.y.view(-1).cpu().detach().numpy()
    with open(path_epo + '.pkl', 'wb') as f:
        pickle.dump({'x': x, 'y': y}, f)

    evaluator = MulticlassEvaluator()
    # acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']
    acc_dic = log_regression(z, data, evaluator, split='preloaded', num_epochs=3000, preload_split=split)
    return acc_dic


@torch.no_grad()
def save_embed(path_epo, model, data, logger):
    model.eval()
    x = model(data.x, data.edge_index)
    x = x.cpu().detach().numpy()
    y = data.y.view(-1).cpu().detach().numpy()
    with open(path_epo + '.pkl', 'wb') as f:
        pickle.dump({'x': x, 'y': y}, f)
    logger.info(path_epo + ' saved.')


def CTAug_node(parser):
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,  # 3000
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    path = 'log/' + args.method + '_' + args.feature + '_' + str(args.seed)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.dataset_file is None:
        if args.dataset == 'Amazon-Computers':
            args.dataset_file = 'data/Amazon-Computers_global_complete_graph_4.pt'
        else:
            args.dataset_file = 'data/' + args.dataset + '_global_complete_graph_5.pt'

    if 'AUG' in args.method:
        path = path + '/' + args.dataset + '_' + args.core + '_' + str(args.factor)
    else:
        path = path + '/' + args.dataset

    logger = config_logger(path + '.log')
    logger.info(args)

    # parse param
    sp = SimpleParam(local_dir='CTAug/methods/node_cls/param', default=default_param)
    if args.dataset == 'Coauthor-CS':
        source = 'local:coauthor_cs.json'
    elif args.dataset == 'Coauthor-Phy':
        source = 'local:coauthor_phy.json'
    elif args.dataset == 'Amazon-Computers':
        source = 'local:amazon_computers.json'
    else:
        raise NotImplementedError

    param = sp(source=source, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    if 'GRACE' in args.method:
        param['drop_scheme'] = 'uniform'

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    if args.feature == 'one':
        dataset = get_dataset('data/' + args.dataset, args.dataset)
        data = dataset[0]
    elif args.feature == 'sub':
        data = torch.load(args.dataset_file)[0]
        data.x = torch.cat((F.normalize(data.identifiers.float(), dim=0), data.x), dim=1)
        id_dim = data.identifiers.shape[1]
        # dataset = [torch.load(args.dataset_file)[0]]
        # dataset, d_id = encode(dataset)
        # attribute = get_dataset_no_norm('data/' + args.dataset, args.dataset)[0].x
        # data = dataset[0]
        # subs = data.identifiers
        # onehots = []
        # for j in range(subs.shape[1]):
        #     onehot = torch.zeros((subs.shape[0], d_id[j]), device=subs.device)
        #     onehot.scatter_(1, subs[:, j:j + 1], 1)
        #     onehots.append(onehot)
        # onehots = torch.cat(onehots, 1)
        # data.x = norm_by_row(torch.cat((onehots, attribute), dim=1))
        # id_dim = np.sum(d_id)
        # print(data, id_dim)
    else:
        raise NotImplementedError

    logger.info(data)
    data = data.to(device)
    data.edge_index = data.edge_index[:, data.edge_index[0, :] != data.edge_index[1, :]]

    if 'AUG' in args.method:
        getcore = GetCore(data, args.core, device, args.factor)
        getcore.preprocess()

    # generate split
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = torch.ones((data.edge_index.size(1),)).to(device)

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    accs, epos, duras, meanss = [], [], [], []

    for t in range(1, 6):
        t0 = time.time()
        if 'OGSN' in args.method:
            encoder = Encoder_OGSN(data.x.shape[1], param['num_hidden'], get_activation(param['activation']),
                                   base_model=get_base_model(param['base_model']), k=param['num_layers'],
                                   id_dim=id_dim).to(device)
        else:
            encoder = Encoder(data.x.shape[1], param['num_hidden'], get_activation(param['activation']),
                              base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)

        model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=param['learning_rate'],
            weight_decay=param['weight_decay']
        )
        means = []
        stds = []
        best_acc = 0
        best_epo = 0
        for epoch in range(1, param['num_epochs'] + 1):
            if args.core == 'kcore':
                core = 'kcore'
            elif args.core == 'ktruss':
                core = 'ktruss'
            else:
                core = random.choice(['kcore', 'ktruss'])

            if 'AUG' in args.method:
                order = random.choice([0, 1, 2])
                weight = getcore.weight[order]
                drop_weights1 = drop_weights * weight
                loss = train(model, optimizer, drop_weights1, args, data, param, feature_weights)
                logger.info(f'(T{t}) | Epoch={epoch:04d}, loss={loss:.4f} | ' + core + str(order))
            else:
                loss = train(model, optimizer, drop_weights, args, data, param, feature_weights)
                logger.info(f'(T{t}) | Epoch={epoch:04d}, loss={loss:.4f}')

            if epoch % 50 == 0:
                path_epo = path + '_' + str(t) + '_' + str(epoch)
                # save_embed(path_epo, model, data, logger)
                acc_dic = test(path_epo, model, data, split)
                acc = acc_dic['mean']
                means.append(acc_dic['mean'])
                stds.append(acc_dic['std'])
                logger.info(f"(E{t}) | Epoch={epoch:04d}, avg_acc={acc_dic['mean']:.2f}±{acc_dic['std']:.2f}")
                if acc > best_acc:
                    best_acc = acc
                    best_epo = epoch

        accs.append(best_acc)
        epos.append(best_epo)
        meanss.append(means)
        logger.info("【Results】")
        for i in range(len(means)):
            logger.info("Epoch: {:}, Accuracy: {:.2f}±{:.2f}".format((i + 1) * 50, means[i], stds[i]))
        logger.info("【Best Accuracy】")
        index = np.argmax(means)
        logger.info("Epoch: {:}, Best Accuracy: {:.2f}±{:.2f}".format((index + 1) * 50, means[index], stds[index]))

        dura = time.time() - t0
        duras.append(dura)
        logger.info("Duration: {:.2f}".format(dura))

    logger.info("【Results】")
    mean = np.mean(meanss, axis=0)
    std = np.std(meanss, axis=0)
    for i in range(len(mean)):
        logger.info("Epoch: {:}, Accuracy: {:.2f}±{:.2f}".format((i + 1) * 50, mean[i], std[i]))

    logger.info("【Final Results】")
    logger.info("Accuracy: {:.2f}±{:.2f}".format(np.mean(accs), np.std(accs)))
    logger.info("Best epoch: {:.2f}±{:.2f}".format(np.mean(epos), np.std(epos)))
    logger.info("Duration: {:.2f}".format(np.mean(duras)))

    logger.info("【Best Accuracy】")
    index = np.argmax(mean)
    logger.info("Epoch: {:}, Best Accuracy: {:.2f}±{:.2f}".format((index + 1) * 50, mean[index], std[index]))

