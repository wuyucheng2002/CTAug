from copy import deepcopy
import GCL.losses as L
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.loader import DataLoader
from ..utils import *
from ..model import *
from ..evaluate import *


def train(gconv, contrast_model, dataloader, optimizer, device, train_gconv, aug_p, getcore, fast=False):
    gconv.train()
    epoch_loss = 0
    step = 0
    step_stop = len(dataloader) // 2 + 1
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)

        if train_gconv:
            xs, edge_indexs, batchs = getcore.drop_joao(data, aug_p)
            data = data.to(device)
            _, g1 = gconv(data.x, data.edge_index, data.batch)
            _, g2 = gconv(xs, edge_indexs, batchs)

            g1, g2 = [gconv.project(g) for g in [g1, g2]]
            loss = contrast_model(g1=g1, g2=g2, batch=data.batch)

            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                xs, edge_indexs, batchs = getcore.drop_joao(data, aug_p)
                data = data.to(device)
                _, g1 = gconv(data.x, data.edge_index, data.batch)
                _, g2 = gconv(xs, edge_indexs, batchs)

                g1, g2 = [gconv.project(g) for g in [g1, g2]]
                loss = contrast_model(g1=g1, g2=g2, batch=data.batch)

        epoch_loss += loss.item()
        step += 1
        if fast:
            if step == step_stop:
                break
    return epoch_loss


def JOAO_AUG(ts, args, path, logger, dataset1, ogsn):
    epo = args.epoch
    batch_size = args.batch_size
    shuffle = args.shuffle
    device = args.device
    pn = args.pn
    eval_model = args.eval_model
    hid_units = args.hid_units
    num_layers = args.num_layer
    interval = args.interval
    save_model = args.save_model
    save_embed = args.save_embed
    factor = args.factor
    norm = args.norm

    dataset, input_dim, id_dim = dataset1
    dataloader_eval = DataLoader(dataset, batch_size=batch_size)

    dataset_train = deepcopy(dataset)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    # print(dataset[0].x.shape)

    duras, accs = [], []

    for t in range(1, ts + 1):
        t0 = time.time()

        if ogsn:
            gconv1 = GConv_OGSN(id_dim=id_dim, input_dim=input_dim, hidden_dim=hid_units,
                                num_layers=num_layers).to(device)
            gconv2 = GConv_OGSN(id_dim=id_dim, input_dim=input_dim, hidden_dim=hid_units,
                                num_layers=num_layers).to(device)
        else:
            gconv1 = GConv(input_dim=input_dim, hidden_dim=hid_units, num_layers=num_layers).to(device)
            gconv2 = GConv(input_dim=input_dim, hidden_dim=hid_units, num_layers=num_layers).to(device)

        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

        optimizer1 = Adam(gconv1.parameters(), lr=0.01)
        optimizer2 = Adam(gconv2.parameters(), lr=0.01)

        getcore1 = GetCore(dataset_train, 'kcore', pn, factor, device)
        getcore2 = GetCore(dataset_train, 'ktruss', pn, factor, device)
        logger.info('Preprocessing is ok.')

        losses = []
        acc, acc_val = 0, 0
        aug_p = np.ones(2) / 2
        for epoch in range(1, epo + 1):
            loss1 = train(gconv1, contrast_model, dataloader_train, optimizer1, device, True, aug_p, getcore1)
            loss2 = train(gconv2, contrast_model, dataloader_train, optimizer2, device, True, aug_p, getcore2)
            loss = loss1 + loss2
            losses.append(loss)

            logger.info("T{}, epoch:{}, loss:{:.4f}, ND: ER = {:.4f}: {:.4f}".format(
                t, epoch, loss, aug_p[0], aug_p[1]))

            if epoch % interval == 0:
                res = test_save_model2(gconv1, gconv2, dataloader_eval, device, path, epoch,
                                       logger, t, norm, save_embed, eval_model, save_model)

                if eval_model:
                    _acc_val, _acc = res
                    if _acc_val > acc_val:
                        acc_val = _acc_val
                        acc = _acc

            # minmax
            if epoch != epo:
                loss_aug = np.zeros(2)
                for n in range(2):
                    _aug_p = np.zeros(2)
                    _aug_p[n] = 1
                    loss_aug[n] = train(gconv1, contrast_model, dataloader_train, optimizer1, device, False,
                                        _aug_p, getcore1) + \
                                  train(gconv2, contrast_model, dataloader_train, optimizer2, device, False,
                                        _aug_p, getcore2)
                gamma = 0.1
                beta = 1
                b = aug_p + beta * (loss_aug - gamma * (aug_p - 1 / 2))

                mu_min, mu_max = b.min() - 1 / 2, b.max() - 1 / 2
                mu = (mu_min + mu_max) / 2
                # bisection method
                while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
                    if np.maximum(b - mu, 0).sum() > 1:
                        mu_min = mu
                    else:
                        mu_max = mu
                    mu = (mu_min + mu_max) / 2

                aug_p = np.maximum(b - mu, 0)
                aug_p /= aug_p.sum()

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