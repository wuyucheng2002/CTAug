from copy import deepcopy
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
    sample = args.sample

    dataset, input_dim, id_dim = dataset1
    dataloader_eval = DataLoader(dataset, batch_size=batch_size)
    # print(dataset[0].x.shape)
    dataset_train = deepcopy(dataset)

    duras, accs = [], []

    for t in range(1, ts + 1):
        t0 = time.time()

        if sample is not None:
            np.random.seed(t)
            sample = min(sample, len(dataset))
            train_list = np.random.permutation(range(len(dataset)))[: sample].tolist()
            # print(train_list)
            dataset_train = [dataset_train[i] for i in train_list]

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)

        if ogsn:
            gconv = GConv_OGSN(id_dim=id_dim, input_dim=input_dim, hidden_dim=hid_units,
                               num_layers=num_layers).to(device)
        else:
            gconv = GConv(input_dim=input_dim, hidden_dim=hid_units, num_layers=num_layers).to(device)

        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
        optimizer = Adam(gconv.parameters(), lr=0.01)
        logger.info('Preprocessing is ok.')

        losses = []
        acc, acc_val = 0, 0
        for epoch in range(1, epo + 1):
            loss = train(gconv, contrast_model, dataloader_train, optimizer, device, pn)
            losses.append(loss)
            logger.info("T{}, epoch:{}, loss:{:.4f}".format(t, epoch, loss))

            if epoch % interval == 0:
                res = test_save_model(gconv, dataloader_eval, device, path, epoch,
                                      logger, t, norm, save_embed, eval_model, save_model)
                if eval_model:
                    _acc_val, _acc = res
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



