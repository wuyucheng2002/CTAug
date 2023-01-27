import random
import GCL.losses as L
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.loader import DataLoader
from ..utils import *
from ..model import *
from ..evaluate import *


def train(gconv, contrast_model, dataloader, optimizer, device, getcore, core1):
    gconv.train()
    epoch_loss = 0
    records = []
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)

        order1 = random.choice([0, 1, 2])
        records.append(core1 + str(order1))

        data = data.to(device)
        xs, edge_indexs, batchs = getcore.get_graph(data, core1, order1)
        _, g1 = gconv(data.x, data.edge_index, data.batch)
        _, g2 = gconv(xs, edge_indexs, batchs)

        g1, g2 = [gconv.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss, records


def GraphCL_AUG(ts, args, path, logger, dataset1, ogsn):
    epo = args.epoch
    batch_size = args.batch_size
    shuffle = args.shuffle
    device = args.device
    hid_units = args.hid_units
    num_layers = args.num_layer
    interval = args.interval
    eval_model = args.eval_model
    save_model = args.save_model
    save_embed = args.save_embed
    factor = args.factor
    pn = args.pn
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

        getcore1 = GetCore(dataset, 'kcore', pn, factor, device)
        getcore2 = GetCore(dataset, 'ktruss', pn, factor, device)

        losses = []
        means, stds = [], []
        for epoch in range(1, epo + 1):
            loss1, records1 = train(gconv1, contrast_model, dataloader, optimizer1, device, getcore1, 'kcore')
            loss2, records2 = train(gconv2, contrast_model, dataloader, optimizer2, device, getcore2, 'ktruss')
            loss = loss1 + loss2
            losses.append(loss)
            logger.info("T{}, epoch:{}, loss:{:.4f}, Property: ".format(t, epoch, loss) +
                        '-'.join(records1) + '-' + '-'.join(records2))

            if epoch % interval == 0 or epoch == epo:
                res = test_save_model2(gconv1, gconv2, dataloader_eval, device, path, epoch,
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
