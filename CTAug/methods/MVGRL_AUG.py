from ..preprocess import *
from ..model import *
from ..evaluate import *


def train(batch_size, model, optimizer, num_nodes, feat, adj, diff, max_nodes, device):
    epoch_loss = 0
    train_idx = np.arange(adj.shape[0])
    np.random.shuffle(train_idx)

    for idx in range(0, len(train_idx), batch_size):
        model.train()
        optimizer.zero_grad()

        batch = train_idx[idx: idx + batch_size]
        mask = num_nodes[idx: idx + batch_size]

        lv1, gv1, lv2, gv2 = model(adj[batch], diff[batch], feat[batch], mask)

        lv1 = lv1.view(batch.shape[0] * max_nodes, -1)
        lv2 = lv2.view(batch.shape[0] * max_nodes, -1)

        batch = torch.LongTensor(np.repeat(np.arange(batch.shape[0]), max_nodes)).to(device)

        loss1 = local_global_loss_(lv1, gv2, batch, 'JSD', mask, device)
        loss2 = local_global_loss_(lv2, gv1, batch, 'JSD', mask, device)
        # loss3 = global_global_loss_(gv1, gv2, 'JSD')
        loss = loss1 + loss2  # + loss3
        loss.backward()
        optimizer.step()

        epoch_loss += loss
    return epoch_loss


def MVGRL_AUG(ts, args, path, logger, dataset1, ogsn):
    hid_units = args.hid_units
    epo = args.epoch
    device = args.device
    batch_size = args.batch_size
    cal_weight = args.cal_weight
    lr = 0.001
    l2_coef = 0.0
    factor = args.factor
    num_layers = args.num_layer
    eval_model = args.eval_model
    interval = args.interval
    save_model = args.save_model
    save_embed = args.save_embed
    norm = args.norm

    graphs, feats, label, id_dim = dataset1
    adj1, diff1, max_nodes = process_graph(graphs, device, factor, 0, (1 - factor), cal_weight)
    adj2, diff2, max_nodes = process_graph(graphs, device, 0, factor, (1 - factor), cal_weight)

    feat_dim = feats[0].shape[-1]
    num_nodes = []
    for idx in range(adj1.shape[0]):
        num_nodes.append(adj1[idx].shape[-1])
        feats[idx] = np.vstack((feats[idx], np.zeros((max_nodes - feats[idx].shape[0], feat_dim))))
    feat = np.array(feats.tolist()).reshape(-1, max_nodes, feat_dim)
    feat = torch.FloatTensor(feat).to(device)
    label = torch.LongTensor(label).to(device)
    ft_size = feat[0].shape[1]
    logger.info('Preprocessing is ok.')

    duras, accs = [], []

    for t in range(1, ts + 1):
        t0 = time.time()

        if ogsn:
            model1 = Model_OGSN(id_dim, ft_size, hid_units, num_layers, device).to(device)
            model2 = Model_OGSN(id_dim, ft_size, hid_units, num_layers, device).to(device)
        else:
            model1 = Model(ft_size, hid_units, num_layers, device).to(device)
            model2 = Model(ft_size, hid_units, num_layers, device).to(device)

        optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=l2_coef)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=l2_coef)

        itr = (adj1.shape[0] // batch_size) + 1

        losses = []
        acc, acc_val = 0, 0
        for epoch in range(1, epo + 1):
            epoch_loss1 = train(batch_size, model1, optimizer1, num_nodes, feat, adj1, diff1, max_nodes, device) / itr
            epoch_loss2 = train(batch_size, model2, optimizer2, num_nodes, feat, adj2, diff2, max_nodes, device) / itr
            loss = epoch_loss1.item() + epoch_loss2.item()
            losses.append(loss)
            logger.info("T{}, epoch:{}, loss:{:.4f}".format(t, epoch, loss))

            if epoch % interval == 0:
                path_epo1 = path + '_' + str(t) + '_' + str(epoch) + '_1'
                path_epo2 = path + '_' + str(t) + '_' + str(epoch) + '_2'
                path_epo = path + '_' + str(t) + '_' + str(epoch)
                if save_model:
                    torch.save(model1.state_dict(), path_epo1 + '_model.pkl')
                    torch.save(model2.state_dict(), path_epo2 + '_model.pkl')

                embed1 = model1.embed(feat, adj1, diff1, num_nodes)
                embed2 = model2.embed(feat, adj2, diff2, num_nodes)
                x1 = embed1.cpu().numpy()
                x2 = embed2.cpu().numpy()
                x = np.concatenate((x1, x2), axis=1)
                y = label.cpu().numpy()

                if save_embed:
                    with open(path_epo + '.pkl', 'wb') as f:
                        pickle.dump({'x': x, 'y': y}, f)
                    logger.info(path_epo + ' saved.')

                if eval_model:
                    _acc_val, _acc = test_SVM(x, y, logger, t, norm)
                    if _acc_val > acc_val:
                        acc_val = _acc_val
                        acc = _acc

        if eval_model:
            logger.info("### Results ###")
            logger.info("Best Accuracy: {:.2f}".format(acc))
            accs.append(acc)

        dura = time.time() - t0
        logger.info("Duration: {:.2f}".format(dura))
        duras.append(dura)

    if eval_model:
        logger.info("### Final Results ###")
        logger.info("Final Best Accuracy: {:.2f}±{:.2f}".format(np.mean(accs), np.std(accs)))
    logger.info("AVG Duration: {:.2f}".format(np.mean(duras)))
    logger.info("AVG Duration/epoch: {:.2f}".format(np.mean(duras)/epo))