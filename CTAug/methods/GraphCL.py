import random
import GCL.losses as L
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.loader import DataLoader
from ..utils import *
from ..model import *
from ..evaluate import *


def train(gconv, contrast_model, dataloader, optimizer, device, pn):
    gconv.train()
    epoch_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        x1, edge_index1, batch1 = drop_edge(data, pn)

        _, g1 = gconv(data.x, data.edge_index, data.batch)
        _, g2 = gconv(x1, edge_index1, batch1)

        g1, g2 = [gconv.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def GraphCL(ts, args, path, logger, dataset1, ogsn):
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
        dataset, input_dim, id_dim = dataset1
        # print(dataset[0].x.shape)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader_eval = DataLoader(dataset, batch_size=batch_size)

        if ogsn:
            gconv = GConv_OGSN(id_dim=id_dim, input_dim=input_dim, hidden_dim=hid_units,
                               num_layers=num_layers).to(device)
        else:
            gconv = GConv(input_dim=input_dim, hidden_dim=hid_units, num_layers=num_layers).to(device)

        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
        optimizer = Adam(gconv.parameters(), lr=0.01)

        losses, means, stds = [], [], []
        for epoch in range(1, epo + 1):
            loss = train(gconv, contrast_model, dataloader, optimizer, device, pn)
            losses.append(loss)
            logger.info("T{}, epoch:{}, loss:{:.4f}".format(t, epoch, loss))

            if epoch % interval == 0 or epoch == epo:
                res = test_save_model(gconv, dataloader_eval, device, path, epoch,
                                      logger, t, norm, save_embed, eval_model, save_model)
                if eval_model:
                    means.append(res[0])
                    stds.append(res[1])

        if eval_model:
            dura = time.time() - t0
            duras.append(dura)
            evaluate(means, stds, dura, logger, interval)
            meanss.append(means)

    if eval_model:
        final_eval2(meanss, logger, interval, duras)



