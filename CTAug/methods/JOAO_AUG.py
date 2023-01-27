import random
import GCL.losses as L
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.loader import DataLoader
from ..utils import *
from ..model import *
from ..evaluate import *


def train(gconv, contrast_model, dataloader, optimizer, device, train_gconv, aug_p, getcore, core1, fast=False):
    gconv.train()
    epoch_loss = 0
    records = []
    step = 0
    step_stop = len(dataloader) // 2 + 1
    for data in dataloader:
        order1 = random.choice([0, 1, 2])
        records.append(core1 + str(order1))

        optimizer.zero_grad()
        data = data.to(device)

        if train_gconv:
            xs, edge_indexs, batchs = getcore.get_graph(data, core1, order1, aug_p)
            data = data.to(device)
            _, g1 = gconv(data.x, data.edge_index, data.batch)
            _, g2 = gconv(xs, edge_indexs, batchs)

            g1, g2 = [gconv.project(g) for g in [g1, g2]]
            loss = contrast_model(g1=g1, g2=g2, batch=data.batch)

            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                xs, edge_indexs, batchs = getcore.get_graph(data, core1, order1, aug_p)
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
    return epoch_loss, records


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

    duras = []
    meanss, meanss1, meanss2 = [], [], []
    for t in range(1, ts + 1):
        t0 = time.time()

        dataset, input_dim, id_dim = dataset1
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader_eval = DataLoader(dataset, batch_size=batch_size)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

        if ogsn:
            gconv1 = GConv_OGSN(id_dim=id_dim, input_dim=input_dim, hidden_dim=hid_units,
                                num_layers=num_layers).to(device)
            gconv2 = GConv_OGSN(id_dim=id_dim, input_dim=input_dim, hidden_dim=hid_units,
                                num_layers=num_layers).to(device)
        else:
            gconv1 = GConv(input_dim=input_dim, hidden_dim=hid_units, num_layers=num_layers).to(device)
            gconv2 = GConv(input_dim=input_dim, hidden_dim=hid_units, num_layers=num_layers).to(device)

        optimizer1 = Adam(gconv1.parameters(), lr=0.01)
        optimizer2 = Adam(gconv2.parameters(), lr=0.01)

        getcore1 = GetCoreJ(dataset, 'kcore', pn, factor, device)
        getcore2 = GetCoreJ(dataset, 'ktruss', pn, factor, device)

        losses = []
        means, stds = [], []
        aug_p = np.ones(2) / 2
        for epoch in range(1, epo + 1):
            loss1, records1 = train(gconv1, contrast_model, dataloader, optimizer1,
                                    device, True, aug_p, getcore1, 'kcore')
            loss2, records2 = train(gconv2, contrast_model, dataloader, optimizer2,
                                    device, True, aug_p, getcore2, 'ktruss')
            loss = loss1 + loss2
            losses.append(loss)

            logger.info("T{}, epoch:{}, loss:{:.4f}, ND: ER = {:.4f}: {:.4f}, Property: ".format(
                t, epoch, loss, aug_p[0], aug_p[1]) + '-'.join(records1) + '-' + '-'.join(records2))

            if epoch % interval == 0 or epoch == epo:
                res = test_save_model2(gconv1, gconv2, dataloader_eval, device, path, epoch,
                                       logger, t, norm, save_embed, eval_model, save_model)

                if eval_model:
                    means.append(res[0])
                    stds.append(res[1])

            # minmax
            if epoch != epo:
                loss_aug = np.zeros(2)
                for n in range(2):
                    _aug_p = np.zeros(2)
                    _aug_p[n] = 1
                    loss_aug[n] = train(gconv1, contrast_model, dataloader, optimizer1, device, False,
                                        _aug_p, getcore1, 'kcore')[0] + \
                                  train(gconv2, contrast_model, dataloader, optimizer2, device, False,
                                        _aug_p, getcore2, 'ktruss')[0]
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
            dura = time.time() - t0
            duras.append(dura)
            evaluate(means, stds, dura, logger, interval)
            meanss.append(means)

    if eval_model:
        final_eval2(meanss, logger, interval, duras)