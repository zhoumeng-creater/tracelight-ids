import argparse
import torch
import numpy as np
import torch.nn.functional as F

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


import os
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import scipy.io as sio
from scipy.optimize import nnls
import random
from torchvision.transforms import Resize
from torchvision import transforms
from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data

import torchvision as tv
from my_attention import build_sliding_attention_mask
from efdmix import EFDMix

def add_imgs(imgs, outdir='tmp.jpg', nrow=10):
    N = imgs.shape
    # if N[1]!=3:
    # imgs = torch.cat([imgs, -1.*imgs[:,:1,:,:]], dim=1)
    imgs = imgs / 2 + 0.5
    imgs = tv.utils.make_grid(imgs, nrow=nrow, pad_value=1)
    tv.utils.save_image(imgs, outdir, nrow=nrow, pad_value=1)

torch_resize = Resize([128, 128])




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from pytorch_transformers import AdamW


alpha=0.1
beta = torch.distributions.Beta(alpha, alpha)
def mix_x(x, y, g, p):
    g, p = g.to(x.device), p.to(x.device)
    x0, x1 = x.chunk(2)
    y0, y1 = y.chunk(2)
    g0, g1 = g.chunk(2)
    p0, p1 = p.chunk(2)

    B, C, W, H = x0.shape
    lmda = np.random.rand(B,) < 0.5
    lmda = torch.tensor(lmda).to(x.device) *1
    # lmda = beta.sample((B, 1, 1, 1))
    # lmda = lmda.to(x.device)

    yout = lmda * y0 + (1-lmda) * y1
    gout = lmda * g0 + (1-lmda) * g1
    pout = lmda * p0 + (1-lmda) * p1
    lmda = lmda.view(-1, 1, 1, 1)
    xout = lmda * x0 + (1 - lmda) * x1

    return xout, yout, gout, pout

def mix_x_0(x, y, g, p):
    g, p = g.to(x.device), p.to(x.device)
    B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
    lmda = np.random.rand(B,) < 0.5
    lmda = torch.tensor(lmda).to(x.device) *1
    # lmda = beta.sample((B, 1, 1, 1))
    # lmda = lmda.to(x.device)
    perm = torch.randperm(B)
    xp = x[perm]
    yp = y[perm]
    gp = g[perm]
    pp = p[perm]

    yout = lmda * y + (1-lmda) * yp
    gout = lmda * g + (1-lmda) * gp
    pout = lmda * p + (1-lmda) * pp
    lmda = lmda.view(-1, 1, 1, 1)
    xout = lmda * x + (1 - lmda) * xp

    x = torch.cat([x, xout], dim=0)
    y = torch.cat([y, yout], dim=0)
    g = torch.cat([g, gout], dim=0)
    p = torch.cat([p, pout], dim=0)

    return x, y, g,p

class BertWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        '''
        have changed line683
        /home/user/.conda/envs/dassl/lib/python3.8/site-packages/pytorch_transformers/modeling_bert.py
        '''
    def forward(self, x, labels=None):
        # tmp_n = x[:, :, 1].sum(-1)
        # new_at = x[:,:,1].unsqueeze(-1).repeat(1,1,300)*0
        # for iii in range(x.shape[0]):
        #     new_at[iii,:tmp_n[iii], :tmp_n[iii]] = build_sliding_attention_mask(tmp_n[iii])
        return self.net(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            # attention_mask=new_at,
            token_type_ids=x[:, :, 2], labels=labels)


def get_optim(net, hparams, net_type=None):
    net_type = 'bert'
    if net_type == 'bert':

        no_decay = ["bias", "LayerNorm.weight"]
        decay_params = []
        nodecay_params = []
        for n, p in net.named_parameters():
            if any(nd in n for nd in no_decay):
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        opt_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": hparams.weight_decay,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(
            opt_grouped_parameters,
            lr=hparams.init_lr,
            betas=hparams.betas #(0.1, 0.999)
            # eps=1e-8
        )

        scheduler = get_scheduler(
            "linear",
            optimizer=opt,
            num_warmup_steps=0,  # 0
            num_training_steps=10,  # 10
        )

        return opt, scheduler


def get_optim_nlp_sgd(net, hparams, net_type=None):
    net_type = 'bert'
    if net_type == 'bert':
        no_decay = ["bias", "LayerNorm.weight"]
        decay_params = []
        nodecay_params = []
        for n, p in net.named_parameters():
            if any(nd in n for nd in no_decay):
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        opt_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": hparams.weight_decay,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]
        opt = torch.optim.SGD(opt_grouped_parameters, lr=hparams.init_lr,
                              momentum=hparams.momentum)

        scheduler = get_scheduler(
            "linear",
            optimizer=opt,
            num_warmup_steps=0,  # 0
            num_training_steps=10,  # 10
        )

        return opt, scheduler


class GeneralizedCELoss(nn.Module):
    """ Generalized Cross Entropy Loss."""

    def __init__(self, q: float = 0.7):
        super(GeneralizedCELoss, self).__init__()
        # q is the hyperparameter for the GCE loss
        self.q = q

    def forward(self, logits, targets):
        """Calculate Loss."""
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise RuntimeError('GCE p is None.')

        y_g = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (y_g.squeeze().detach() ** self.q) * self.q
        if np.isnan(y_g.mean().item()):
            raise RuntimeError('GCE y_g is None')
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss.mean()


def which_fc(d, neck, n_classes):
    fc = nn.Sequential(
        torch.nn.Linear(d, neck),
        nn.ReLU(),
        torch.nn.Linear(neck, d),
        nn.ReLU(),
        torch.nn.Linear(d, n_classes)
    )
    # fc = nn.Linear(d, n_classes)
    # init_fc(fc)
    return fc


def moving_average(model, shadow, alpha=0.99):
    with torch.no_grad():
        for param, shadow_param in zip(model.parameters(), shadow.parameters()):
            shadow_param.data = alpha * shadow_param.data + (1.-alpha) * param.data
        for bn, shadow_bn in zip(model.modules(), shadow.modules()):
            if isinstance(bn, nn.BatchNorm2d):
                shadow_bn.running_mean.data = alpha * shadow_bn.running_mean.data + (1.-alpha) * bn.running_mean.data
                shadow_bn.running_var.data = alpha * shadow_bn.running_var.data + (1.-alpha) * bn.running_var.data


def balance_half(y_count, ratio=0.1):
    c_num = torch.tensor(y_count)
    half = c_num.sum() / c_num
    half = half / half.max() * ratio
    return half


def init_fc(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# def group_weight_specific(y_class0, y_place0, n_classes=None):
#     y_class = y_place0
#     y_place = y_class0
#
#     mat = torch.eye(n_classes)
#     group = []
#     for i in range(n_classes):
#         # if i == 4:
#         #     print(i)
#         y_i = y_place[y_class == i]
#         y_hot = mat[y_i]
#         group.append(y_hot.sum(dim=0, keepdim=True))
#     group = torch.cat(group, dim=0)
#     N_m = group.sum(dim=0, keepdim=True)
#     prob = N_m / (group + 1e-5)* (group>0)
#
#     tmp1 = prob/prob.sum(dim=1, keepdim=True)
#     weight = tmp1/(group+1e-5) * (group>0)
#     weight_v = weight.view(-1)
#
#     group_v = y_class * n_classes + y_place
#     weight_sample = weight_v[group_v]
#     return weight_sample
def group_weight_specific(y_place, y_class, n_classes=None):

    mat = torch.eye(n_classes)
    group = []
    for i in range(n_classes):
        # if i == 4:
        #     print(i)
        y_i = y_class[y_place == i]
        y_hot = mat[y_i]
        group.append(y_hot.sum(dim=0, keepdim=True))
    group = torch.cat(group, dim=0)
    N_m = group.sum(dim=0, keepdim=True)
    prob = N_m / (group + 1e-5)* (group>0)

    tmp1 = prob/prob.sum(dim=-1, keepdim=True)
    weight = tmp1/(group+1e-5) * (group>0)
    weight_v = weight.view(-1)

    group_v = y_place * n_classes + y_class
    weight_sample = weight_v[group_v]
    return weight_sample

def group_weight_specific_v4_good(y_place, y_class, n_classes=None):
    mat = torch.eye(n_classes)
    group = []
    for i in range(n_classes):
        # if i == 4:
        #     print(i)
        y_i = y_class[y_place == i]
        y_hot = mat[y_i]
        group.append(y_hot.sum(dim=0, keepdim=True))
    group = torch.cat(group, dim=0)
    N_m = group.sum(dim=1, keepdim=True)
    prob = N_m / (group + 1e-5)* (group>0)
    # prob = prob / prob.sum(dim=0, keepdim=True)
    weight = prob / (group + 1e-5) * (group > 0)
    weight_v = weight.view(-1)

    group_v = y_place * n_classes + y_class
    weight_sample = weight_v[group_v]
    return weight_sample


def group_weight_specific_v4_good4cifar(y_place, y_class, n_classes=None):
    mat = torch.eye(n_classes)
    group = []
    for i in range(n_classes):
        y_i = y_class[y_place == i]
        y_hot = mat[y_i]
        group.append(y_hot.sum(dim=0, keepdim=True))
    group = torch.cat(group, dim=0)
    group_p = group / group.sum()
    p_m = group_p.sum(dim=1, keepdim=True)
    p_c = group / group.sum(dim=0, keepdim=True)
    prob = p_m / (p_c + 1e-5)* (group>0)
    # prob = prob / prob.sum(dim=0, keepdim=True)
    weight = prob / (group + 1e-5) * (group > 0)
    weight_v = weight.view(-1)

    group_v = y_place * n_classes + y_class
    weight_sample = weight_v[group_v]
    return weight_sample


def group_weight_specific_v4(y_place, y_class, n_classes=None):
    mat = torch.eye(n_classes)
    group = []
    for i in range(n_classes):
        y_i = y_class[y_place == i]
        y_hot = mat[y_i]
        group.append(y_hot.sum(dim=0, keepdim=True))
    group = torch.cat(group, dim=0)
    group_p0 = group / group.sum()
    # sio.savemat('cifar10-pct5-G.mat',{'G':group_p0.data.cpu().numpy()})
    # sio.savemat('CA-G.mat',{'G':group_p0.data.cpu().numpy()})
    ''' class balance '''
    tmp = group.sum(dim=0, keepdim=True)
    lamd = tmp.max()/tmp
    groupB = group*lamd
    group_p = groupB / groupB.sum()
    p_m = group_p.sum(dim=1, keepdim=True)
    p_c = group_p / group_p.sum(dim=0, keepdim=True)
    prob = p_m / (p_c + 1e-5)* (group>0)
    weight_g = prob * lamd
    # weight = prob*1

    weight = weight_g / (group + 1e-5) * (group > 0)
    weight_v = weight.view(-1)
    # weight_v = weight.view(-1) + 1e-5
    # weight_v = weight.view(-1).pow(1.8)

    group_v = y_place * n_classes + y_class
    weight_sample = weight_v[group_v]
    print('group_weight = ', np.bincount(group_v))
    # return weight_sample, weight_v
    return weight_sample, weight_g, weight_v


def group_weight_specific_v44(y_place, y_class, n_classes=None):
    mat = torch.eye(n_classes)
    group = []
    for i in range(n_classes):
        # if i == 4:
        #     print(i)
        y_i = y_class[y_place == i]
        y_hot = mat[y_i]
        group.append(y_hot.sum(dim=0, keepdim=True))
    group = torch.cat(group, dim=0)
    A = group.max()/(group + 1e-5)* (group > 0)
    ''' solve: A.t() * lambda = 1 '''
    b = torch.ones(n_classes)
    # lamb = torch.linalg.solve(A.t(), b)
    # lamb, residual = nnls(A.t(), b)
    A_n = torch.pinverse(A.t())
    lamb = A_n.mm(b.view(-1,1))

    prob = A.t() * lamb.t()
    weight = prob.t() / (group + 1e-5) * (group > 0)
    weight_v = weight.view(-1)

    group_v = y_place * n_classes + y_class
    weight_sample = weight_v[group_v]
    return weight_sample


def group_weight_specific_v1(y_place, y_class, n_classes=None):
    y_p = (y_place != y_class) * 1
    mat = torch.eye(2)
    group = []
    for i in range(n_classes):
        # if i == 4:
        #     print(i)
        y_i = y_p[y_place == i]
        y_hot = mat[y_i]
        group.append(y_hot.sum(dim=0, keepdim=True))
    group = torch.cat(group, dim=0)
    N_m = group.sum(dim=1, keepdim=True)
    prob = N_m / (group + 1e-5)* (group>0)

    tmp1 = prob/prob.sum(dim=1, keepdim=True)
    weight = tmp1/(group+1e-5) * (group>0)
    weight_v = weight.view(-1)

    group_v = y_place * 2 + y_p
    weight_sample = weight_v[group_v]
    return weight_sample


def which_data(dname, basedir=None, test_dir=None, half=None, sub_idx=None,update_g=False):
    if dname == 'cCIFAR10':
        from dataset_cifar import ImageLabelFilelist as imageread
        target_resolution = (128, 128)
        train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=True, scale=1.)
        test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False, scale=1.)

        trainset = imageread(basedir, stage='train', transform=test_transform, half=half, sub_idx=sub_idx,update_g=update_g)
        testset_dict = {
            'wb': imageread(test_dir, stage='test', transform=test_transform),
            'wb_val': imageread(basedir, stage='val', transform=test_transform),
        }
    elif dname == 'cMNIST':
        from dataset_cmnist import ImageLabelFilelist as imageread
        input_size = 28
        test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        ])

        trainset = imageread(basedir, stage='train', transform=test_transform, half=half, sub_idx=sub_idx,update_g=update_g)
        testset_dict = {
            'wb': imageread(test_dir, stage='test', transform=test_transform),
            'wb_val': imageread(basedir, stage='val', transform=test_transform),
        }
    else:
        target_resolution = (224, 224)
        train_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=True, scale=1.)
        test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False, scale=1.)

        trainset = WaterBirdsDataset(basedir=basedir, split="train", transform=test_transform, half=half, sub_idx=sub_idx)
        testset_dict = {
            'wb': WaterBirdsDataset(basedir=test_dir, split="test", transform=test_transform),
            'wb_val': WaterBirdsDataset(basedir=test_dir, split="val", transform=test_transform),
        }
    return trainset, testset_dict


def covariate(z):
    z_mean = z.mean(dim=0, keepdim=True)
    z_ = z - z_mean
    z_norm = z_.norm(dim=0,keepdim=True)
    z_n = z_/z_norm
    C = z_n.T.mm(z_n)
    mask = torch.eye(z.shape[1]).to(z.device)
    csum = (C*(1-mask)).abs().sum()
    return C, csum


def which_data2(dname, insize=None, basedir=None, test_dir=None, half=None, sub_idx=None, update_g=True):
    # update_g = True # False
    if dname == 'cCIFAR10':
        from dataset_cifar import ImageLabelFilelist as imageread
        target_resolution = (128, 128)
        train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=True,scale=1.)
        test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False,scale=1.)

        trainset = imageread(basedir, stage='train', transform=test_transform, half=half, sub_idx=sub_idx, update_g=update_g)
        valset = imageread(basedir, stage='val', transform=test_transform)
        testset = imageread(test_dir, stage='test', transform=test_transform)
    elif dname == 'catdog':
        from dataset_catdog import ImageLabelFilelist as imageread
        target_resolution = (128, 128)
        train_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=True, scale=1.)
        test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False, scale=1.)

        trainset = imageread(basedir, split="train", transform=test_transform, half=half, sub_idx=sub_idx)
        valset = imageread(basedir, split="val", transform=test_transform)
        testset = imageread(basedir, split="test", transform=test_transform)
    elif dname == 'UTKFace':
        from dataset_UTKFace import UTKFace as imageread
        target_resolution = (128, 128)
        train_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=True, scale=1.)
        test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False, scale=1.)

        trainset = imageread(basedir, split="train", transform=test_transform, half=half, sub_idx=sub_idx)
        valset = imageread(basedir, split="val", transform=test_transform)
        testset = imageread(basedir, split="test", transform=test_transform)
    elif dname == 'cMNIST':
        from dataset_cmnist import ImageLabelFilelist as imageread
        if insize is None:
            input_size = 28
        else:
            input_size = insize
        test_transform  = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        ])

        trainset = imageread(basedir, stage='train', transform=test_transform, half=half, sub_idx=sub_idx, update_g=update_g)
        valset = imageread(basedir, stage='val', transform=test_transform)
        testset = imageread(test_dir, stage='test', transform=test_transform)
    elif dname =='CivilComments':
        from dataset_civil_org import ImageLabelFilelist as imageread
        trainset = imageread(img_file=basedir,split='train', half=half, sub_idx=sub_idx)
        valset = imageread(img_file=basedir,split="val", half=half)
        testset = imageread(img_file=basedir,split="test", half=half)
    elif dname =='MultiNLI':
        from dataset_multi import ImageLabelFilelist as imageread
        trainset = imageread(basedir+'train_data.mat', half=half, sub_idx=sub_idx)
        valset = imageread(basedir+'test_data.mat')
        testset = imageread(basedir+'val_data.mat')
    else:
        if insize is None:
            # target_resolution = (128,128)
            target_resolution = (224,224)
        else:
            target_resolution = insize
        train_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=True, scale=1.)
        test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False, scale=1.)

        # target_resolution = (128,128)
        target_resolution = (224, 224)
        trainset = WaterBirdsDataset(basedir=basedir, split="train", transform=test_transform, half=half, sub_idx=sub_idx, dname=dname)
        valset = WaterBirdsDataset(basedir=test_dir, split="val", transform=test_transform,dname=dname)
        testset = WaterBirdsDataset(basedir=test_dir, split="test", transform=test_transform, dname=dname)

    return trainset, valset, testset


''' for cCIFAR, WB and CELEBA '''
def get_embed(m, x, layer=None):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    if layer==1:
        fea = x.mean([2,3])
    x = m.layer2(x)
    if layer==2:
        fea = x.mean([2,3])
    x = m.layer3(x)
    if layer==3:
        fea = x.mean([2,3])
    x = m.layer4(x)

    x = m.avgpool(x)
    fea0 = torch.flatten(x, 1)
    if layer is None:
        fea = fea0
        # fea_list = [torch.histogram(row.cpu(), bins=128, range=(0.,5.), density=True)[0].view(1,-1) for row in fea0]
        # fea = torch.cat(fea_list)
    logi = m.fc(fea0)
    pred = logi.argmax(-1)
    return fea, logi, pred


def feat(model, dataloader, dname=None, vod=0):
    model.eval()
    all_fea, all_x, all_y, all_py, all_idx, all_g = [], [], [], [], [], []
    for x, y, g, p in dataloader:
        with torch.no_grad():
            if dname in ['CivilComments', 'MultiNLI']:
                tmp = model(x.cuda(), y.cuda())
                all_fea.append(tmp[2].detach().cpu().numpy())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_py.append(tmp[1].argmax(-1).detach().cpu().numpy())
            else:
                tmp = get_embed(model, x.cuda())
                if vod==1:
                    xv, x_i = tmp[0].detach().sort(dim=1)
                    xv = xv / xv.norm(p=2, dim=1, keepdim=True) *3.0
                    all_fea.append(xv.cpu().numpy())
                elif vod==2:
                    xv, x_i = tmp[0].detach().sort(dim=1)
                    x_i = x_i.argsort(-1)
                    xo = x_i / x_i.max()
                    all_fea.append(xo.cpu().numpy())
                elif vod == 3:
                    xv, x_i = tmp[0].detach().sort(dim=1)
                    xv = xv / xv.norm(p=2, dim=1, keepdim=True) * 3.0
                    x_i = x_i.argsort(-1)
                    xo = x_i / x_i.max()
                    x_all = torch.cat([xo, xv], dim=-1)
                    all_fea.append(x_all.cpu().numpy())
                else:
                    all_fea.append(tmp[0].detach().cpu().numpy())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_py.append(tmp[2].detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
            all_g.append(g.detach().cpu().numpy())
            all_idx.append(p.detach().cpu().numpy())
    all_x = np.vstack(all_x)
    all_fea = np.vstack(all_fea)
    all_y = np.concatenate(all_y)
    all_py = np.concatenate(all_py)
    all_g = np.concatenate(all_g)
    all_idx = np.concatenate(all_idx)
    model.train()
    return all_x, all_y, all_g, all_idx,all_py, all_fea


def feat_pair(model, dataloader, dname=None):
    model.eval()
    all_fea, all_x, all_y, all_py, all_idx, all_g, all_fea0 = [], [], [], [], [], [], []
    for x0, y0, g0, x1, y1, g1, _ in dataloader:
        x = torch.cat([x0, x1], dim=0).cuda()
        y = torch.cat([y0, y1], dim=0).cuda()
        g = torch.cat([g0, g1], dim=0)
        with torch.no_grad():
            if dname in ['CivilComments', 'MultiNLI']:
                tmp = model(x0.cuda(), y0.cuda())
                all_fea.append(tmp[2].detach().cpu().numpy())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_py.append(tmp[1].argmax(-1).detach().cpu().numpy())
            else:
                tmp = model(x, is_train=True, is_fea=True)
                all_fea.append(tmp[0].chunk(2)[0].detach().cpu().numpy())
                all_x.append(tmp[1].chunk(2)[0].detach().cpu().numpy())
                all_py.append(tmp[2].chunk(2)[0].detach().cpu().numpy())
                # tmp = model(x, is_train=False, is_fea=True)
                # all_fea0.append(tmp[0].chunk(2)[0].detach().cpu().numpy())
            all_y.append(y0.detach().cpu().numpy())
            all_g.append(g0.detach().cpu().numpy())
    all_x = np.vstack(all_x)
    all_fea = np.vstack(all_fea)
    # all_fea0 = np.vstack(all_fea0)
    # all_fea = np.vstack([all_fea0, all_fea])
    all_y = np.concatenate(all_y)
    # all_y = np.concatenate([all_y, all_y+2])
    all_py = np.concatenate(all_py)
    all_g = np.concatenate(all_g)
    model.train()
    return all_x, all_y, all_g, all_g,all_py, all_fea


def feat_hist(model, dataloader, dname=None, d_hist=128):
    model.eval()
    all_fea, all_x, all_y, all_py, all_idx, all_g = [], [], [], [], [], []
    for x, y, g, p in dataloader:
        with torch.no_grad():
            if dname in ['CivilComments', 'MultiNLI']:
                tmp = model(x.cuda(), y.cuda())
                all_fea.append(tmp[2].detach().cpu().numpy())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_py.append(tmp[1].argmax(-1).detach().cpu().numpy())
            else:
                tmp = get_embed(model, x.cuda())
                fea_list = [torch.histogram(row.cpu(), bins=d_hist, range=(0.,5.), density=True)[0].view(1,-1) for row in tmp[0].detach()]
                fea = torch.cat(fea_list)
                logi = model.fc_hist(fea.cuda())
                pred = logi.argmax(-1)

                all_fea.append(tmp[0].detach().cpu().numpy())
                all_x.append(logi.detach().cpu().numpy())
                all_py.append(pred.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
            all_g.append(g.detach().cpu().numpy())
            all_idx.append(p.detach().cpu().numpy())
    all_x = np.vstack(all_x)
    all_fea = np.vstack(all_fea)
    all_y = np.concatenate(all_y)
    all_py = np.concatenate(all_py)
    all_g = np.concatenate(all_g)
    all_idx = np.concatenate(all_idx)
    model.train()
    return all_x, all_y, all_g, all_idx,all_py, all_fea


def feat_style(model, dataloader, dname=None, layer=None):
    model.eval()
    all_fea, all_x, all_y, all_py, all_idx, all_g = [], [], [], [], [], []
    for x, y, g, p in dataloader:
        with torch.no_grad():
            if dname in ['CivilComments', 'MultiNLI']:
                tmp = model(x.cuda(), y.cuda())
                all_fea.append(tmp[2].detach().cpu().numpy())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_py.append(tmp[1].argmax(-1).detach().cpu().numpy())
            else:
                tmp = get_embed(model, x.cuda(), layer=layer)
                all_fea.append(tmp[0].detach().cpu().numpy())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_py.append(tmp[2].detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
            all_g.append(g.detach().cpu().numpy())
            all_idx.append(p.detach().cpu().numpy())
    all_x = np.vstack(all_x)
    all_fea = np.vstack(all_fea)
    all_y = np.concatenate(all_y)
    all_py = np.concatenate(all_py)
    all_g = np.concatenate(all_g)
    all_idx = np.concatenate(all_idx)
    model.train()
    return all_x, all_y, all_g, all_idx,all_py, all_fea


''' for cMNIST '''
class MLP_ula(nn.Module):
    # n_outputs = 256
    n_outputs = 100
    def __init__(
            self, in_dim, h_dim: int=100, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            )
        self.fc = nn.Linear(h_dim, num_classes)
    def forward(self, x):
        fea = self.net(x.flatten(1))
        out = self.fc(fea)
        return out

    def get_embed(self, x):
        fea = self.net(x.flatten(1))
        logi = self.fc(fea)
        pred = logi.argmax(-1)
        return fea, logi, pred

    def feat(self, dataloader):
        all_x, all_y, all_py, all_idx, all_g = [], [], [], [], []
        for x, y, g, p in dataloader:
            with torch.no_grad():
                tmp = self.get_embed(x.cuda())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
                all_py.append(tmp[2].detach().cpu().numpy())
                all_g.append(g.detach().cpu().numpy())
                all_idx.append(p.detach().cpu().numpy())
        all_x = np.vstack(all_x)
        all_y = np.concatenate(all_y)
        all_py = np.concatenate(all_py)
        all_g = np.concatenate(all_g)
        all_idx = np.concatenate(all_idx)
        return all_x, all_y, all_g, all_idx, all_py


class MNIST_CNN_fc0(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape, num_classes=10, neck=10):
        super(MNIST_CNN_fc0, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, padding=1)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(128, num_classes)
        # self.fc = nn.Sequential(
        #     torch.nn.Linear(128, neck, bias=False),
        #     torch.nn.Linear(neck, 128),
        #     nn.ReLU(),
        #     torch.nn.Linear(128, num_classes)
        # )

    def forward(self, x, use_clip=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        fea = x.mean(dim=(2, 3))
        if use_clip:
            fea = torch.clip(fea, -3., 3.)

        fea = self.dropout(fea)
        x = self.fc(fea)
        return x

    def get_embed(self, x):
        # self.eval()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        fea = x.mean(dim=(2, 3))

        logi = self.fc(fea)
        pred = logi.argmax(-1)
        # self.train()
        return fea, logi, pred

    def feat(self, dataloader):
        # self.eval()
        all_x, all_y, all_py, all_idx, all_g, all_fea = [], [], [], [], [], []
        for x, y, g, p in dataloader:
            # for ff in dataloader:
            with torch.no_grad():
                tmp = self.get_embed(x.cuda())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_fea.append(tmp[0].detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
                all_py.append(tmp[2].detach().cpu().numpy())
                all_g.append(g.detach().cpu().numpy())
                all_idx.append(p.detach().cpu().numpy())
        all_x = np.vstack(all_x)
        all_fea = np.vstack(all_fea)
        all_y = np.concatenate(all_y)
        all_py = np.concatenate(all_py)
        all_g = np.concatenate(all_g)
        all_idx = np.concatenate(all_idx)
        # self.train()
        return all_x, all_y, all_g, all_idx, all_py, all_fea


class MNIST_CNN_fc_org(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape, num_classes=10, neck=10):
        super(MNIST_CNN_fc, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64,  128, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, 2, padding=1)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        # self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(256, num_classes)
        # self.fc = nn.Sequential(
        #     torch.nn.Linear(128, neck, bias=False),
        #     torch.nn.Linear(neck, 128),
        #     nn.ReLU(),
        #     torch.nn.Linear(128, num_classes)
        # )

    def forward(self, x, use_clip=False):
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)

        fea = x.mean(dim=(2, 3))
        if use_clip:
            fea = torch.clip(fea, -3., 3.)

        # fea = self.dropout(fea)
        x = self.fc(fea)
        return x

    def get_embed(self, x):
        # self.eval()
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)

        fea = x.mean(dim=(2, 3))

        logi = self.fc(fea)
        pred = logi.argmax(-1)
        # self.train()
        return fea, logi, pred

    def feat(self, dataloader):
        # self.eval()
        all_x, all_y, all_py, all_idx, all_g, all_fea = [], [], [], [], [], []
        for x, y, g, p in dataloader:
            # for ff in dataloader:
            with torch.no_grad():
                tmp = self.get_embed(x.cuda())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_fea.append(tmp[0].detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
                all_py.append(tmp[2].detach().cpu().numpy())
                all_g.append(g.detach().cpu().numpy())
                all_idx.append(p.detach().cpu().numpy())
        all_x = np.vstack(all_x)
        all_fea = np.vstack(all_fea)
        all_y = np.concatenate(all_y)
        all_py = np.concatenate(all_py)
        all_g = np.concatenate(all_g)
        all_idx = np.concatenate(all_idx)
        # self.train()
        return all_x, all_y, all_g, all_idx, all_py, all_fea


class MNIST_CNN_fc(nn.Module):
    def __init__(self, input_shape, num_classes=10, neck=10):
        super(MNIST_CNN_fc, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64,  128, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, 2, padding=1)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        # self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, use_clip=False):
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)

        fea = x.mean(dim=(2, 3))
        if use_clip:
            fea = torch.clip(fea, -3., 3.)

        # fea = self.dropout(fea)
        x = self.fc(fea)
        return x

    def get_embed(self, x):
        # self.eval()
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)

        fea = x.mean(dim=(2, 3))

        logi = self.fc(fea)
        pred = logi.argmax(-1)
        # self.train()
        return fea, logi, pred

    def feat(self, dataloader, vod=0):
        # self.eval()
        all_x, all_y, all_py, all_idx, all_g, all_fea = [], [], [], [], [], []
        for x, y, g, p in dataloader:
            # for ff in dataloader:
            with torch.no_grad():
                tmp = self.get_embed(x.cuda())
                if vod==1:
                    xv, x_i = tmp[0].detach().sort(dim=1)
                    xv = xv / xv.norm(p=2, dim=1, keepdim=True) *3.0
                    all_fea.append(xv.cpu().numpy())
                elif vod==2:
                    xv, x_i = tmp[0].detach().sort(dim=1)
                    x_i = x_i.argsort(-1)
                    xo = x_i / x_i.max() * 3.
                    xo = xo / xo.norm(p=2, dim=1, keepdim=True)
                    all_fea.append(xo.cpu().numpy())
                elif vod==3:
                    xv, x_i = tmp[0].detach().sort(dim=1)
                    xv = xv / xv.norm(p=2, dim=1, keepdim=True) * 3.0
                    x_i = x_i.argsort(-1)
                    xo = x_i / x_i.max() * 3.
                    x_all = torch.cat([xo, xv], dim=-1)
                    all_fea.append(x_all.cpu().numpy())
                else:
                    all_fea.append(tmp[0].detach().cpu().numpy())

                all_x.append(tmp[1].detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
                all_py.append(tmp[2].detach().cpu().numpy())
                all_g.append(g.detach().cpu().numpy())
                all_idx.append(p.detach().cpu().numpy())
        all_x = np.vstack(all_x)
        all_fea = np.vstack(all_fea)
        all_y = np.concatenate(all_y)
        all_py = np.concatenate(all_py)
        all_g = np.concatenate(all_g)
        all_idx = np.concatenate(all_idx)
        # self.train()
        return all_x, all_y, all_g, all_idx, all_py, all_fea


class MNIST_CNN_fc_efdm(nn.Module):
    n_outputs = 128

    def __init__(self, input_shape, num_classes=10, neck=10):
        super(MNIST_CNN_fc_efdm, self).__init__()
        # self.mix_layers=['layer1', 'layer2', 'layer3']
        self.mix_layers=['layer1']
        self.mixstyle_p=1.0
        self.mixstyle_alpha=0.1
        self.efdmix = EFDMix(p=self.mixstyle_p, alpha=self.mixstyle_alpha)

        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, padding=1)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, is_train=False, is_fea=False, use_clip=False):
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)
        if is_train and 'layer1'in self.mix_layers:
            x = self.efdmix(x)[0]

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        if is_train and 'layer2'in self.mix_layers:
            x = self.efdmix(x)[0]

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        if is_train and 'layer3'in self.mix_layers:
            x = self.efdmix(x)[0]

        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)
        if is_train and 'layer4'in self.mix_layers:
            x = self.efdmix(x)[0]

        # fea = x.mean(dim=(2, 3))
        fea = self.avgpool(x)
        fea = torch.flatten(fea, 1)

        if use_clip:
            fea = torch.clip(fea, -3., 3.)

        # fea = self.dropout(fea)
        x = self.fc(fea)
        if is_fea:
            logi = x
            pred = logi.argmax(-1)
            return fea, logi, pred
        else:
            return x

    def get_embed(self, x):
        # self.eval()
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)

        fea = x.mean(dim=(2, 3))

        logi = self.fc(fea)
        pred = logi.argmax(-1)
        # self.train()
        return fea, logi, pred

    def feat(self, dataloader):
        # self.eval()
        all_x, all_y, all_py, all_idx, all_g, all_fea = [], [], [], [], [], []
        for x, y, g, p in dataloader:
            # for ff in dataloader:
            with torch.no_grad():
                tmp = self.get_embed(x.cuda())
                all_x.append(tmp[1].detach().cpu().numpy())
                all_fea.append(tmp[0].detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
                all_py.append(tmp[2].detach().cpu().numpy())
                all_g.append(g.detach().cpu().numpy())
                all_idx.append(p.detach().cpu().numpy())
        all_x = np.vstack(all_x)
        all_fea = np.vstack(all_fea)
        all_y = np.concatenate(all_y)
        all_py = np.concatenate(all_py)
        all_g = np.concatenate(all_g)
        all_idx = np.concatenate(all_idx)
        # self.train()
        return all_x, all_y, all_g, all_idx, all_py, all_fea


def loss_center_my_v0(fea, y, weight):

    fea_m = fea - fea.mean(0, keepdim=True).data
    U, S, V = fea_m.svd()
    tmp = fea_m.mm(V)
    X_emb = tmp[:, :20]

    # X_emb = fea

    center_all = X_emb.mean(0, keepdim=True).data
    n_y = y.max()+1
    c = torch.zeros(n_y, X_emb.shape[1]).to(fea.device)
    xw = X_emb * (weight.view(-1, 1))
    for i in range(n_y):
        c[i,:] = xw[y==i,:].sum(dim=0, keepdim=True)

    loss = (c - center_all).abs().sum()
    # loss = (c - c.mean(0, keepdim=True).data).abs().sum()

    return loss



def rotate_img(x, y):
    '''
    x: input images
    y: the rotation to be applied to x
    '''
    N = x.shape[0]
    x_r = torch.zeros_like(x).to(x.device)
    for i in range(N):
        # if y[i] == 0, no rotate, return the original img
        if y[i] == 1: # left90
            tmp = x[i,...].permute(0,2,1)
            x_r[i, ...] = tmp.flip(1)
        elif y[i] == 2: # left180
            tmp = x[i,...].flip(1)
            x_r[i, ...] = tmp.flip(2)
        elif y[i] == 3: # left270
            tmp = x[i,...].permute(0,2,1)
            x_r[i, ...] = tmp.flip(2)
        elif y[i] == 4: # 0, flip
            x_r[i, ...] = x[i, ...].flip(2)
        elif y[i] == 5: # left90, flip
            x_r[i,...] = x[i,...].permute(0,2,1)
        elif y[i] == 6: # left180, flip
            x_r[i,...] = x[i,...].flip(1)
        elif y[i] == 7: # left270, flip
            tmp = x[i,...].permute(0,2,1)
            tmp = tmp.flip(1)
            x_r[i, ...] = tmp.flip(2)
        else:
            x_r[i, ...] = x[i, ...]+0.

    return x_r


def loss_center_my(fea, y, weight):

    fea_m = fea - fea.mean(0, keepdim=True).data
    U, S, V = fea_m.svd()
    tmp = fea_m.mm(V)
    X_emb = tmp[:, :10]

    # X_emb = fea

    center_all = X_emb.mean(0, keepdim=True).data
    n_y = y.max()+1
    c = torch.zeros(n_y, X_emb.shape[1]).to(fea.device)
    for i in range(n_y):
        c[i,:] = X_emb[y==i,:].mean(dim=0, keepdim=True)
    center_all1 = c.mean(dim=0, keepdim=True)

    xw = X_emb * (weight.view(-1, 1))
    for i in range(n_y):
        c[i,:] = xw[y==i,:].sum(dim=0, keepdim=True)

    loss = (c - center_all1.data).abs().sum()
    # loss = (c - c.mean(0, keepdim=True).data).abs().sum()

    return loss

# def cov_sqrt_loss(Ci, C):
#     eps = 1e-5* torch.eye(C.shape[0]).to(C.device)
#     Csqr = torch.linalg.cholesky(C +eps)
#     tmp = Csqr.mm(Ci).mm(Csqr)
#     tmp_sqr = torch.linalg.cholesky(tmp +eps)
#     loss = torch.trace(Ci) + torch.trace(C) - 2.*torch.trace(tmp_sqr)
#     return loss

def sqrtm(C):
    U, S, V = torch.svd(C)
    S_sqrt = torch.sqrt(S)
    return U @ torch.diag(S_sqrt) @ V.T

def cov_sqrt_loss(Ci, C):
    # eps = 1e-5* torch.eye(C.shape[0]).to(C.device)
    Csqr = sqrtm(C)
    tmp = Csqr @ Ci @ Csqr
    tmp_sqr = sqrtm(tmp)
    loss = torch.trace(Ci) + torch.trace(C) - 2.*torch.trace(tmp_sqr)
    return loss


def loss_w2_my(fea, y, weight):
    NN, DD = fea.shape
    fea_m = fea - fea.mean(0, keepdim=True).data
    U, S, V = fea_m.svd()
    tmp = fea_m.mm(V)
    X_emb = tmp[:, :10]
    NN, DD = X_emb.shape
    # X_emb = fea

    # center_all = X_emb.mean(0, keepdim=True).data

    n_y = y.max()+1
    c = torch.zeros(n_y, X_emb.shape[1]).to(fea.device)
    for i in range(n_y):
        c[i,:] = X_emb[y==i,:].mean(dim=0, keepdim=True)
    center_all1 = c.mean(dim=0, keepdim=True)

    x_rm = X_emb - center_all1
    cov_all = (x_rm.t().mm(x_rm))/NN

    c = torch.zeros(n_y, X_emb.shape[1]).to(fea.device)
    cov_c = torch.zeros(n_y, DD, DD).to(fea.device)
    loss = 0.
    for i in range(n_y):
        tmp_x = X_emb[y==i,:]
        tmp_w = weight[y==i].view(-1, 1)

        c[i, :] = (tmp_x * tmp_w).sum(dim=0, keepdim=True)
        loss1 = (c[i, :] - center_all1.data).abs().sum()
        # loss1 = (c[i, :] - center_all1.data).pow(2).sum()

        tmp_x_rm = (tmp_x-tmp_x.mean(0, keepdim=True)).data
        cov_c[i] = (tmp_w * tmp_x_rm).t().mm(tmp_x_rm)
        loss2 = cov_sqrt_loss(cov_c[i], cov_all.detach())

        # loss += (loss1 + 100.*loss2)
        loss += (loss1 + loss2)

    return loss


def loss_covariate_my(fea, weight=None):
    # fea = fea[:,:-2]
    HH, WW = fea.shape
    x_mean = fea.mean(0, keepdim=True)
    xp = fea - x_mean

    xw = xp
    cov = torch.matmul(xw.t(), xw*weight.view(-1, 1))
    mask = torch.eye(fea.shape[1]).to(fea.device)
    Cdiag = cov.diag().mean()
    cov = cov/Cdiag
    # loss = loss.abs().mean()
    loss = (cov*(1.-mask)).abs().sum()/(WW*WW-WW)
    # loss = loss.abs().sum()

    return loss


def weight_logit2prob(w_l, yc, det=1.0):
    n_y = yc.max() + 1
    soft_w = w_l.data*1.
    for i in range(n_y):
        tmp = w_l[yc == i]
        # tmp = F.softmax(tmp * 2., dim=0)# 2,1,0.5 pct
        tmp = F.softmax(tmp * det, dim=0)#5pct
        soft_w[yc == i] = tmp
    return soft_w


class Net_weight(nn.Module):
    def __init__(self, num_sample=None, ini=None, smooth_m=None):
        super(Net_weight, self).__init__()

        self.num_sample = num_sample
        self.w_para = nn.Parameter(ini)
        self.softmax = nn.Softmax(0)
        self.num_f = 10
        self.sum = True
        self.smooth_m = smooth_m

    def forward(self, x, y=None, device=None):
        ''' original linear + nonlinear independent '''
        # self.soft_w = self.softmax(self.w_para)
        # lossb = loss_expect.lossb_expect(x, self.soft_w, self.num_f, self.sum)
        ''' my linear independent '''
        n_y = y.max() + 1
        self.soft_w = self.w_para *1.
        for i in range(n_y):
            tmp = self.w_para[y==i] * 1.
            tmp = self.softmax(tmp)
            self.soft_w[y==i] = tmp

        lossg = loss_covariate_my(x, self.soft_w)

        ''' my equal spectrum '''
        # self.soft_w = self.softmax(self.w_para) + 1/len(self.w_para)
        # lossb = loss_expect.loss_spectrum_my(x, self.soft_w)
        # # lossp = soft_w.pow(2).sum()
        # # lossp = -(self.soft_w.log() * self.soft_w).sum()
        # lossg = lossb
        # # loss_s = loss_expect.smooth_my(self.smooth_m.data, self.soft_w)
        # # lossg = lossb + 100.*loss_s
        ''' my equal center '''
        # n_y = y.max() + 1
        # self.soft_w = self.w_para *1.
        # for i in range(n_y):
        #     tmp = self.w_para[y==i] * 1.
        #     tmp = self.softmax(tmp)
        #     self.soft_w[y==i] = tmp
        #
        # lossb = loss_w2_my(x, y, self.soft_w)
        # lossg = lossb #+ 0.1*lossp

        ''' my equal center '''
        # n_y = y.max() + 1
        # self.soft_w = self.w_para *1.
        # for i in range(n_y):
        #     tmp = self.w_para[y==i] * 1.
        #     tmp = self.softmax(tmp)
        #     self.soft_w[y==i] = tmp
        # lossg = loss_expect.loss_distribution(x,y,self.soft_w)

        return lossg

def compute_fea(envs, model, device=None, dataname=None):
    model.eval()
    fea_all, y_all, ind_all = [], [], []
    for dataloadi in envs:
        for data, label, idx1, idx2 in dataloadi:
            idx = idx2 if dataname=='CUB' else idx1
            with torch.no_grad():
                x = torch_resize(data.data.to(device))
                fea = model(x)[1]
                fea_all.append(fea)
                y_all.append(label)
                ind_all.append(idx)

    fea_all = torch.cat(fea_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    ind_all = torch.cat(ind_all, dim=0)
    return fea_all, y_all, ind_all


def plt_class(X_emb, y, lim_axis, save_dir, w=None):
    fig, ax = plt.subplots(figsize=(6,6))
    margins = {"left": 0.10, "bottom": 0.10, "right": 1. - 0.10, "top": 1. - 0.10}
    fig.subplots_adjust(**margins)
    leg = ['%d'%i for i in range(y.max()+1)]
    # leg = ['%d'%i for i in range(10)]
    y = y.squeeze()
    # plt.figure(dpi=150)
    N = y.max()+1
    num = torch.zeros(N,)
    for i in range(N):
        num[i] = (y==i).sum()
    # if is_sort:
    #     sort_v, sort_i = num.sort(descending=True)
    # else:
    #     sort_i =

    for jj in range(N):
        alpha = 0.9
        if w is None:
            ind = (y == jj)
        else:
            ind = (y == jj) * (w>= 0.001)
        ax.scatter(X_emb[ind, 0], X_emb[ind, 1], s=5., label=leg[jj], alpha=alpha, edgecolors='none')
        # ax.scatter(X_emb[y == jj, 0], X_emb[y == jj, 1], s=10., alpha=0.3, edgecolors='none')
    plt.axis(lim_axis)
    legend = ax.legend(loc='best', frameon=False, ncol=2)
    frame = legend.get_frame()
    frame.set_facecolor('none')
    # ax.grid(True)
    plt.savefig(save_dir, dpi=600)
    plt.close()

def plt_class_center(X_emb, y, lim_axis, save_dir):
    num_class = y.max() + 1
    # cmap = matplotlib.cm.get_cmap('Paired', num_class)
    cmap = matplotlib.colormaps.get_cmap('Paired')
    colors = cmap(np.linspace(0,1,num_class))
    fig, ax = plt.subplots()
    margins = {"left": 0.05, "bottom": 0.05, "right": 1. - 0.00, "top": 1. - 0.00}
    fig.subplots_adjust(**margins)
    leg = ['%d' % i for i in range(num_class)]
    y = y.squeeze()
    # plt.figure(dpi=150)
    for jj in range(num_class):
        tmp_x = X_emb[y == jj, ...]
        ax.scatter(tmp_x[0, 0], tmp_x[0, 1], s=25., label=leg[jj], color=colors[jj], alpha=1, edgecolors='none', marker='p')
        ax.scatter(tmp_x[1:, 0], tmp_x[1:, 1], s=5., label=leg[jj], color=colors[jj], alpha=0.4, edgecolors='none')
    plt.axis(lim_axis)
    # legend = ax.legend(loc='best', frameon=False, ncol=2)
    # frame = legend.get_frame()
    # frame.set_facecolor('none')
    # ax.grid(True)
    plt.savefig(save_dir, dpi=200)
    plt.close()

def plt_each_class_center(X_emb, y, lim_axis, shots=0, save_dir=None):
    num_class = y.max() + 1
    cmap = matplotlib.cm.get_cmap('Paired', num_class)
    leg = ['%d' % i for i in range(num_class)]
    y = y.squeeze()
    margins = {"left": 0.05, "bottom": 0.05, "right": 1. - 0.00, "top": 1. - 0.00}

    for jj in range(len(leg)):
        fig, ax = plt.subplots()
        fig.subplots_adjust(**margins)
        # plt.figure(dpi=150)
        tmp_x = X_emb[y == jj, ...]
        ax.scatter(tmp_x[0, 0], tmp_x[0, 1], s=25., label=leg[jj], c='r', alpha=1, edgecolors='none', marker='p')
        ax.scatter(tmp_x[1:shots+1, 0], tmp_x[1:shots+1, 1], s=10., label=leg[jj], c='m', alpha=0.7, edgecolors='none', marker='p')

        ax.scatter(tmp_x[shots+1:, 0], tmp_x[shots+1:, 1], s=10., label=leg[jj], c='b', alpha=0.4, edgecolors='none')
        tmp_x = X_emb[:num_class, ...]
        ax.scatter(tmp_x[:, 0], tmp_x[:, 1], s=15., label=leg[jj], c='r', alpha=1., edgecolors='none')

        plt.axis(lim_axis)
        legend = ax.legend(loc='best', frameon=False, ncol=2)
        frame = legend.get_frame()
        frame.set_facecolor('none')
        # ax.grid(True)
        plt.savefig(save_dir[:-4]+'_%s.jpg'%leg[jj], dpi=200)
        plt.close()



def plot_PCA(fea, yc, yd=None, w=None, name=''):
    fea_m = fea - fea.mean(0, keepdim=True).data

    U, S, V = fea_m.svd()
    tmp = fea_m.mm(V)
    X_emb = tmp[:, :2].cpu().numpy()
    det = 0.01
    lim_axis = [X_emb.min(0)[0] - det, X_emb.max(0)[0] + det, X_emb.min(0)[1] - det, X_emb.max(0)[1] + det]

    plt_class(X_emb, yc.cpu().numpy(), lim_axis,
              save_dir='./tmp_fig/%s_pca_class.jpg'%name)


def plot_tsne(fea, yg, ypg=None, w=None, name=''):
    X_emb = TSNE(n_components=2).fit_transform(fea)
    det = 0.1
    lim_axis = [X_emb.min(0)[0] - det, X_emb.max(0)[0] + det, X_emb.min(0)[1] - det, X_emb.max(0)[1] + det]
    plt_class(X_emb, yg, lim_axis,
              save_dir='./tmp_fig/%s_gt_y.png'%name)
    plt_class(X_emb, ypg, lim_axis,
              save_dir='./tmp_fig/%s_p_y.png' % name)



def feature(model, device, test_loader):
    model.eval()
    f_softmax = nn.Softmax(1)
    test_loss = 0
    correct = 0
    fea, yc, yd, w_gt, y_pred, ids = [], [], [], [], [], []
    with torch.no_grad():
        for s_id, data, target, d, w in test_loader:
            data, target, d, w = data.to(device), target.to(device), d.to(device), w.to(device)
            output, tmp = model(data)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            y_pred_tmp = f_softmax(output) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            fea.append(tmp)
            yc.append(target)
            yd.append(d)
            w_gt.append(w)
            y_pred.append(y_pred_tmp)
            ids.append(s_id)

    fea = torch.cat(fea,0)
    yc = torch.cat(yc, 0)
    yd = torch.cat(yd, 0)
    w_gt = torch.cat(w_gt, 0)
    y_pred = torch.cat(y_pred, 0)
    ids = torch.cat(ids, 0)
    # though the accuracy=100%, there are errors
    y_pred = y_pred[ids,:]

    return fea, yc, yd, w_gt, y_pred


def feature_vae(model, device, test_loader):
    model.eval()
    f_softmax = nn.Softmax(1)
    test_loss = 0
    correct = 0
    fea, yc, yd, w_gt, y_pred, ids = [], [], [], [], [], []
    with torch.no_grad():
        for s_id, data, target, d, w in test_loader:
            data, target, d, w = data.to(device), target.to(device), d.to(device), w.to(device)
            rec, z = model(data)[:2]
            y_pred_tmp = (rec - data).abs().sum([1,2,3]) # get the index of the max log-probability

            fea.append(z)
            yc.append(target)
            yd.append(d)
            w_gt.append(w)
            y_pred.append(y_pred_tmp)
            ids.append(s_id)

    fea = torch.cat(fea,0)
    yc = torch.cat(yc, 0)
    yd = torch.cat(yd, 0)
    w_gt = torch.cat(w_gt, 0)
    y_pred = torch.cat(y_pred, 0)
    ids = torch.cat(ids, 0)
    # though the accuracy=100%, there are errors
    y_pred = y_pred[ids]

    return fea, yc, yd, w_gt, y_pred


def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True


def reweight_comput(dataset_train, model, device):
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1000, shuffle=False, \
                                               num_workers=4, drop_last=False)
    fea, yc, yd, w_gt, y_pred = feature(model, device, train_loader)
    info = [y_pred[i, yc[i]] for i in range(len(yc))]
    weight = 1. + (yc.max()+1)/len(yc) - torch.tensor(info).to(device)
    # weight = 1. + 1/len(yc) - torch.tensor(info).to(device)

    # fea = fea - fea.mean(-1, keepdim=True)
    # CC = fea.t().mm(fea)
    # UU, SS, VV = CC.svd()
    # info = SS.cumsum(dim=0) / SS.sum()
    # select_dim = VV[:, info>0.9]
    # fea_map = fea.mm(select_dim)
    # dist = fea_map.pow(2).sum(-1).sqrt()
    # weight = (dist/dist.max()).pow(2)

    return weight, fea, yc, yd, w_gt


def reweight_comput_vae(dataset_train, model, device):
    softmax = nn.Softmax(0)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1000, shuffle=False, \
                                               num_workers=4, drop_last=False)
    fea, yc, yd, w_gt, y_pred = feature_vae(model, device, train_loader)
    # info = [y_pred[i, yc[i]] for i in range(len(yc))]
    # weight = 1. + (yc.max()+1)/len(yc) - torch.tensor(info).to(device)
    # weight = y_pred
    weight = softmax(y_pred)*0.5 + 0.5

    return weight, fea, yc, yd, w_gt


def main():
    # DATASET ='oxford_flowers'
    # DATASET ='DTD'
    DATASET = 'EuroSAT'
    # DATASET = 'StanfordCars'
    # DATASET = 'FGVCAircraft'
    # DATASET = 'UCF101'
    # DATASET ='OxfordPets'
    # DATASET ='caltech101' # 93.5%,
    # DATASET ='fgvc_aircraft'
    # DATASET ='stanford_cars'
    # DATASET ='ucf101'
    # DATASET ='oxford_pets'
    # DATASET ='sun397'
    # DATASET ='food101'
    # DATASET ='imagenet'

    our=2
    ''' OUR '''
    if our==1:
        data = sio.loadmat('feature_our_%s.mat'%DATASET)
        fea_all, text_features, gamma,beta,yt = torch.from_numpy(data['fea_all']),torch.from_numpy(data['text_features']), \
            torch.from_numpy(data['gamma_all']),torch.from_numpy(data['beta_all']),torch.from_numpy(data['yt_all'])

        num_class = yt.max()+1
        out_dir = '/home/user/workspace/TaskRes-main-1/TEMP/'
        os.makedirs(out_dir) if not os.path.isdir(out_dir) else None
        y_text = torch.linspace(0,num_class-1,steps=num_class).long().view(-1, 1).numpy()

        # fea_all = torch.cat([text_features, fea_all], dim=0)
        # yall = np.concatenate((y_text, yt.cpu().view(-1, 1).numpy()), axis=0)

        yall = yt.cpu().view(-1, 1).numpy()

        fea_all = fea_all
        X_emb = TSNE(n_components=2).fit_transform(fea_all.cpu().numpy())
        lim_axis = [X_emb.min(0)[0] - 1, X_emb.max(0)[0] + 1, X_emb.min(0)[1] - 1, X_emb.max(0)[1] + 1]
        plt_class(X_emb, yall, lim_axis,
                  save_dir=out_dir + '/tsne_%s_our_all.jpg' % (DATASET))

        fea_all = gamma
        X_emb = TSNE(n_components=2).fit_transform(fea_all.cpu().numpy())
        lim_axis = [X_emb.min(0)[0] - 1, X_emb.max(0)[0] + 1, X_emb.min(0)[1] - 1, X_emb.max(0)[1] + 1]
        plt_class(X_emb, yall, lim_axis,
                  save_dir=out_dir + '/tsne_%s_our_gamma_all.jpg' % (DATASET))

        fea_all = beta
        X_emb = TSNE(n_components=2).fit_transform(fea_all.cpu().numpy())
        lim_axis = [X_emb.min(0)[0] - 1, X_emb.max(0)[0] + 1, X_emb.min(0)[1] - 1, X_emb.max(0)[1] + 1]
        plt_class(X_emb, yall, lim_axis,
                  save_dir=out_dir + '/tsne_%s_our_beta_all.jpg' % (DATASET))
    elif our==2:
        ''' CLIP-Adapter '''
        data = sio.loadmat('feature_clip_adapter_%s.mat' % DATASET)
        fea_all, text_features, yt = torch.from_numpy(data['fea_all']), torch.from_numpy(
            data['text_features']), torch.from_numpy(data['yt_all'])

        num_class = yt.max() + 1
        out_dir = '/home/user/workspace/TaskRes-main-1/TEMP/'
        os.makedirs(out_dir) if not os.path.isdir(out_dir) else None
        y_text = torch.linspace(0, num_class - 1, steps=num_class).long().view(-1, 1).numpy()

        # fea_all = fea_all
        fea_all = fea_all / fea_all.norm(dim=-1, keepdim=True)
        yall = yt.cpu().view(-1, 1).numpy()

        X_emb = TSNE(n_components=2).fit_transform(fea_all.cpu().numpy())
        lim_axis = [X_emb.min(0)[0] - 1, X_emb.max(0)[0] + 1, X_emb.min(0)[1] - 1, X_emb.max(0)[1] + 1]

        plt_class(X_emb, yall, lim_axis,
                  save_dir=out_dir + '/tsne_%s_clip_adapter.jpg' % (DATASET))
    else:
        ''' ZERO '''
        data = sio.loadmat('feature_zero_%s.mat' % DATASET)
        fea_all, text_features, yt = torch.from_numpy(data['fea_all']), torch.from_numpy(
            data['text_features']), torch.from_numpy(data['yt_all'])

        num_class = yt.max() + 1
        out_dir = '/home/user/workspace/TaskRes-main-1/TEMP/'
        os.makedirs(out_dir) if not os.path.isdir(out_dir) else None
        y_text = torch.linspace(0, num_class - 1, steps=num_class).long().view(-1, 1).numpy()

        # fea_all = fea_all
        fea_all = fea_all / fea_all.norm(dim=-1, keepdim=True)
        yall = yt.cpu().view(-1, 1).numpy()

        X_emb = TSNE(n_components=2).fit_transform(fea_all.cpu().numpy())
        lim_axis = [X_emb.min(0)[0] - 1, X_emb.max(0)[0] + 1, X_emb.min(0)[1] - 1, X_emb.max(0)[1] + 1]

        plt_class(X_emb, yall, lim_axis,
                  save_dir=out_dir + '/tsne_%s_zero_all.jpg' % (DATASET))
    print('finished')



if __name__ == "__main__":

    main()
    '''
    train_dataloder, drop_last=Flase, line44:/home/user/Desktop/Dassl.pytorch/dassl/data/data_manager.py
    /home/user/Desktop/Dassl.pytorch/dassl/engine/trainer.py zhong gai def build_data_loader
    CUDA_VISIBLE_DEVICES=0 python train_debug_rl.py
    
    '''
