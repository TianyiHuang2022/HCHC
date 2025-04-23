from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import matplotlib.pyplot as pyplot
from utils import fashionDataset
from sklearn.decomposition import PCA
from utils_algo import PairEnum, BCE_softlabels, cluster_acc


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.sparse_(m.weight.data, sparsity=0.9)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

        self.apply(weight_init)

    def forward(self, x):

        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path='data/ae_fashion.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', path)

    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        center = self.cluster_layer
        return x_bar, q, center


def add_noise(img):
	noise = torch.randn(img.size()) * 0.2
	noisy_img = img + noise
	return noisy_img

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def CKROD(Dist, sigma):
    Dist = Dist.cpu()
    Dist = Dist / Dist.max()
    Rank = Dist.argsort().argsort()
    Rdist = Rank + Rank.t()+1
    #Rdist = Rdist / Rdist.max()
    KROD = Rdist.float() * torch.exp(Dist / sigma)
    return KROD


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(0):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))



def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train_idec():

    model = IDEC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)

    #  model.pretrain('data/ae_mnist.pkl')
    model.pretrain()

    bce = BCE_softlabels()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)
    # cluster parameter initiate
    data = dataset.x
    y = dataset.y



    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, n_init='auto')
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    bate = 5
    m = 0.8
    for epoch in range(20):
        #adjust_learning_rate(optimizer, epoch)
        if epoch % args.update_interval == 0:

            _, tmp_q,_ = model(data)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
        for batch_idx, (x, _, idx) in enumerate(train_loader):



            xn = add_noise(x)
            xn = xn.to(device)
            x = x.to(device)
            idx = idx.to(device)

            x_bar, q, _ = model(x)
            xn_bar, qn, _ = model(xn)
            _, hidden = model.ae(x)

            feat_detach = hidden.detach()
            feat_row, feat_col = PairEnum(feat_detach)
            tmp_distance_ori = ((feat_row - feat_col) ** 2.).sum(1).view(x.size(0), x.size(0))
            tmp_distance_ori = CKROD(tmp_distance_ori, 10)
            tmp_distance_ori = tmp_distance_ori.cpu().float()
            target_ulb = torch.zeros_like(tmp_distance_ori).float()
            target_ulb[tmp_distance_ori < torch.kthvalue(tmp_distance_ori, 12, 0, True)[0]] = 1
            target_ulb = target_ulb.mul(torch.exp(-0.2*tmp_distance_ori))
            target_ulb = target_ulb.view(-1)

            prob_bottleneck_row, prob_bottleneck_col = PairEnum(q)
            m_loss = bce(prob_bottleneck_row, prob_bottleneck_col, target_ulb)
            m_loss = max(bate*m**epoch,0.2)*m_loss
            m_loss = m_loss.cuda()




            reconstr_loss = F.mse_loss(x_bar, x)
            lossc = F.mse_loss(qn, q)
            idx = idx.long()
            kl_loss = F.kl_div(q.log(), p[idx],reduction='batchmean')
            loss = reconstr_loss + m_loss + lossc * 10 + 0.001 * kl_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    output_f = tmp_q.cpu().numpy()
    np.save('embedded_fashion.npy', output_f)
    _, hidden = model.ae(data)
    hidden = hidden.cpu()
    hidden = hidden.detach().numpy()
    _x_bar, _q, center = model(data)
    center = center.cpu()
    center = center.detach().numpy()
    np.save('embedded_fashionz.npy', hidden)
    np.save('center_fashion.npy', center)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_z', default=5, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--pretrain_path', type=str, default='data/ae_fashion.pkl')
    parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.n_input = 784
    dataset = fashionDataset()
    Y = dataset.y
    np.save('fashion_y.npy', Y)


    print(args)
    train_idec()
