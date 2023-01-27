from GCL.models import SingleBranchContrast
import GCL.losses as L
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from ..model import *
from ..evaluate import *


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)


def train(encoder_model, contrast_model, dataloader, optimizer, device):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None or data.x.shape[1] == 0:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        loss = contrast_model(h=z, g=g, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def InfoGraph(ts, args, path, logger, dataset1):
    epo = args.epoch
    batch_size = args.batch_size
    shuffle = args.shuffle
    device = args.device
    hid_units = args.hid_units
    num_layers = args.num_layer
    interval = args.interval
    save_model = args.save_model
    save_embed = args.save_embed
    norm = args.norm
    eval_model = args.eval_model

    duras, meanss = [], []

    for t in range(1, ts + 1):
        t0 = time.time()

        dataset, input_dim, _ = dataset1

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader_eval = DataLoader(dataset, batch_size=batch_size)
        gconv = GConv(input_dim=input_dim, hidden_dim=hid_units,
                      activation=torch.nn.ReLU, num_layers=num_layers).to(device)
        fc1 = FC(hidden_dim=hid_units * 2)
        fc2 = FC(hidden_dim=hid_units * 2)
        encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
        contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)
        optimizer = Adam(encoder_model.parameters(), lr=0.01)

        losses, means, stds = [], [], []
        for epoch in range(1, epo + 1):
            loss = train(encoder_model, contrast_model, dataloader, optimizer, device)
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
