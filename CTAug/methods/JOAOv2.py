import random
import GCL.losses as L
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.loader import DataLoader
from ..utils import *
from ..model import *
from ..evaluate import *


def train(gconv, contrast_model, dataloader, optimizer, device, train_gconv, aug_p, pn, n, fast=False):
    gconv.train()
    epoch_loss = 0
    step = 0
    step_stop = len(dataloader) // 3 + 1
    for data in dataloader:
        optimizer.zero_grad()
        aug = get_aug_p(data, aug_p, pn)

        if train_gconv:
            _, edge_index1, _ = aug(data.x, data.edge_index)
            data = data.to(device)
            edge_index1 = edge_index1.to(device)

            _, g1 = gconv(data.x, data.edge_index, data.batch)
            _, g2 = gconv(data.x, edge_index1, data.batch)
            g1, g2 = [gconv.project[n](g) for g in [g1, g2]]
            loss = contrast_model(g1=g1, g2=g2, batch=data.batch)

            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                _, edge_index1, _ = aug(data.x, data.edge_index)
                data = data.to(device)
                edge_index1 = edge_index1.to(device)

                _, g1 = gconv(data.x, data.edge_index, data.batch)
                _, g2 = gconv(data.x, edge_index1, data.batch)
                g1, g2 = [gconv.project[n](g) for g in [g1, g2]]
                loss = contrast_model(g1=g1, g2=g2, batch=data.batch)

        epoch_loss += loss.item()
        step += 1
        if fast:
            if step == step_stop:
                break
    return epoch_loss


def JOAOv2(ts, args, path, logger, dataset1):
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
    norm = args.norm

    duras, meanss = [], []

    for t in range(1, ts + 1):
        t0 = time.time()

        dataset, input_dim, _ = dataset1
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader_eval = DataLoader(dataset, batch_size=batch_size)
        gconv = GConv_proj(input_dim=input_dim, hidden_dim=hid_units, num_layers=num_layers).to(device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
        optimizer = Adam(gconv.parameters(), lr=0.01)

        losses, means, stds = [], [], []
        aug_p = np.ones(3) / 3
        for epoch in range(1, epo + 1):
            loss = train(gconv, contrast_model, dataloader, optimizer, device, True, aug_p, pn, 0)
            losses.append(loss)
            logger.info("T{}, epoch:{}, loss:{:.4f}, ND: ER: SUB = {:.4f}: {:.4f}: {:.4f}".format(
                t, epoch, loss, aug_p[0], aug_p[1], aug_p[2]))

            if epoch % interval == 0 or epoch == epo:
                res = test_save_model(gconv, dataloader_eval, device, path, epoch,
                                      logger, t, norm, save_embed, eval_model, save_model)
                if eval_model:
                    means.append(res[0])
                    stds.append(res[1])

            # minmax
            if epoch != epo:
                loss_aug = np.zeros(3)
                for n in range(3):
                    _aug_p = np.zeros(3)
                    _aug_p[n] = 1
                    loss_aug[n] = train(gconv, contrast_model, dataloader, optimizer, device, False,
                                        _aug_p, pn, n+1)

                gamma = 0.1
                beta = 1
                b = aug_p + beta * (loss_aug - gamma * (aug_p - 1 / 3))

                mu_min, mu_max = b.min() - 1 / 3, b.max() - 1 / 3
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
