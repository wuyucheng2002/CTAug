from ..preprocess import *
from ..model import *
from ..evaluate import *


def train(batch_size, model, optimizer, num_nodes, feat, adj, diff, max_nodes, device):
    epoch_loss = 0.0
    train_idx = np.arange(len(adj))
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
        epoch_loss += loss
        loss.backward()
        optimizer.step()

    return epoch_loss


def MVGRL(ts, args, path, logger, dataset1, ogsn):
    hid_units = args.hid_units
    epo = args.epoch
    device = args.device
    batch_size = args.batch_size
    lr = 0.001
    l2_coef = 0.0
    num_layers = args.num_layer
    interval = args.interval
    eval_model = args.eval_model
    save_model = args.save_model
    save_embed = args.save_embed
    norm = args.norm

    duras, meanss = [], []

    for t in range(1, ts + 1):
        t0 = time.time()
        graphs, feats, label, id_dim = dataset1
        adj, diff, max_nodes = process_graph(graphs, device)

        feat_dim = feats[0].shape[-1]
        num_nodes = []
        for idx in range(adj.shape[0]):
            num_nodes.append(adj[idx].shape[-1])
            feats[idx] = np.vstack((feats[idx], np.zeros((max_nodes - feats[idx].shape[0], feat_dim))))
        feat = np.array(feats.tolist()).reshape(-1, max_nodes, feat_dim)
        feat = torch.FloatTensor(feat).to(device)
        label = torch.LongTensor(label).to(device)
        ft_size = feat[0].shape[1]

        if ogsn:
            model = Model_GSN(id_dim, ft_size, hid_units, num_layers, device).to(device)
        else:
            model = Model(ft_size, hid_units, num_layers, device).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        itr = (adj.shape[0] // batch_size) + 1

        losses, means, stds = [], [], []
        for epoch in range(1, epo + 1):
            epoch_loss = train(batch_size, model, optimizer, num_nodes, feat, adj, diff, max_nodes, device) / itr
            losses.append(epoch_loss.item())
            logger.info('T{}, Epoch:{}, Loss:{:.4f}'.format(t, epoch, epoch_loss))

            if epoch % interval == 0 or epoch == epo:
                path_epo = path + '_' + str(t) + '_' + str(epoch)
                if save_model:
                    torch.save(model.state_dict(), path_epo + '_model.pkl')

                embed = model.embed(feat, adj, diff, num_nodes)
                x = embed.cpu().numpy()
                y = label.cpu().numpy()

                if save_embed:
                    with open(path_epo + '.pkl', 'wb') as f:
                        pickle.dump({'x': x, 'y': y}, f)
                    logger.info(path_epo + ' saved.')

                if eval_model:
                    mean, std = test_LR(x, y, logger, t, norm)
                    means.append(mean)
                    stds.append(std)

        if eval_model:
            dura = time.time() - t0
            duras.append(dura)
            evaluate(means, stds, dura, logger, interval)
            meanss.append(means)

    if eval_model:
        final_eval2(meanss, logger, interval, duras)


