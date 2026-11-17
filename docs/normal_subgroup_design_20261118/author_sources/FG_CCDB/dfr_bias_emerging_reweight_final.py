"""Evaluate DFR on spurious correlations datasets."""
import copy

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from functools import partial
import pickle
from torchvision import transforms
from fast_data_loader import InfiniteDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from show_tsne import plot_tsne, get_embed, feat, which_data, which_data2, add_imgs, MNIST_CNN_fc, group_weight_specific, group_weight_specific_v4
criterion = torch.nn.CrossEntropyLoss(reduce=False)


C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
REG = "l1"
CLASS_WEIGHT_OPTIONS = [1.0]
CLASS_WEIGHT_OPTIONS = [{0: 1., 1: w} for w in CLASS_WEIGHT_OPTIONS]

cw_4_cifar = {}
for i in range(10):
    cw_4_cifar[i]=1.


parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
parser.add_argument(
    "--data_dir", type=str,
    default=None,
    help="Train dataset directory")
parser.add_argument(
    "--result_path", type=str, default="logs/",
    help="Path to save results")
parser.add_argument(
    "--ckpt_path", type=str, default=None, help="Checkpoint path")
parser.add_argument(
    "--batch_size", type=int, default=300, required=False,
    help="Checkpoint path")
parser.add_argument(
    "--balance_dfr_val", type=bool, default=True, required=False,
    help="Subset validation to have equal groups for DFR(Val)")
parser.add_argument(
    "--notrain_dfr_val", type=bool, default=True, required=False,
    help="Do not add train data for DFR(Val)")
parser.add_argument(
    "--tune_class_weights_dfr_train", action='store_true',
    help="Learn class weights for DFR(Train)")
args = parser.parse_args()


def dfr_on_validation_eval(
        train_loader, pseudo_g, test_loader, val_loader, valpseudo_g, valall_idx, num_retrains=2000,
        preprocess=True, dname=None, n_groups=None, weight=None, train_loaders_1=None, gt_g=None):
    coefs, intercepts = [], []

    # for iter in range(50):
    best_val, best_test, best_val_g, best_test_g = -1., 0., 0., 0.
    iter=0

    for iter, data in enumerate(train_loader):
    # for x_train, y_train, g_train, p_train in next(train_loader):

        x_train = data[0]
        y_train = data[1].view(-1)
        g_train = data[2].view(-1)
        idx = data[3].view(-1)
        # print(np.bincount(g_train))
        # if train_loaders_1 is not None:
        #     # n_g = np.bincount(g_train)
        #     sub_psudo_g = pseudo_g[idx]
        #     n_g = np.bincount(sub_psudo_g)
        #     data_1 = next(train_loaders_bl)
        #     need_x = data_1[0].squeeze()[n_g==0]
        #     need_y = data_1[1].squeeze().view(-1)[n_g==0]
        #     need_g = data_1[2].squeeze().view(-1)[n_g==0]
        #     x_train = torch.cat([x_train,need_x])
        #     y_train = torch.cat([y_train,need_y])
        #     g_train = torch.cat([g_train,need_g])

        # # add_imgs(x_train, 'tmp.jpg')
        # w_train = weight[p_train].cuda()

        optimizer.zero_grad()
        out = model(x_train.cuda())
        loss = criterion(out, y_train.cuda())

        w_sub = weight[idx].cuda()**args.exp_lam
        loss = loss * w_sub

        loss = loss.mean()
        # loss = (loss*w_train).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iter > num_retrains:
            break
        if iter%100 == 0:
            model.eval()
            y_train = y_train.numpy()
            with torch.no_grad():
                # preds_test = model.get_embed(x_test)[-1]
                # preds_train = model.get_embed(x_train)[-1]
                if dname == 'cMNIST':
                    tmpp = model.feat(val_loader)
                    preds_val, y_val, g_val = tmpp[4], tmpp[1], tmpp[2]
                    tmpp = model.feat(test_loader)
                    preds_test, y_test, g_test = tmpp[4], tmpp[1], tmpp[2]
                    # preds_train = model.get_embed(x_train.cuda())[2].data.cpu().numpy()
                else:
                    _, y_val, g_val, _, preds_val, _ = feat(model, val_loader)
                    _, y_test, g_test, _, preds_test, _ = feat(model, test_loader)
                    # preds_train = get_embed(model, x_train.cuda())[2].data.cpu().numpy()

            if dname in ['cCIFAR10', 'cMNIST']:
                g_val = y_val # valall_idx,
                n_groups_val = np.bincount(valpseudo_g)
                # print('n_groups_val ========= ', n_groups_val)
                n_groups_val_min = n_groups_val[n_groups_val>0].min()
                test_accs = [(preds_test == y_test)[y_test == g].mean()
                             for g in range(n_classes)]# g_test[0-19]
                # val_accs = [(preds_val == y_val)[y_val == g][:n_groups_val_min].mean()
                #               for g in range(n_classes)]
                val_accs = [(preds_val == y_val)[y_val == g][:n_groups_val_min].mean()
                              for g in range(n_classes)]
            else:
                g_val = valpseudo_g # valall_idx,
                n_groups_val = np.bincount(valpseudo_g)
                n_groups_val_min = n_groups_val.min()
                test_accs = [(preds_test == y_test)[g_test == g].mean()
                             for g in range(n_groups)]
                val_accs = [(preds_val == y_val)[g_val == g][:n_groups_val_min].mean()
                              for g in range(n_groups)]
            test_mean_acc = (preds_test == y_test).mean()
            test_wors_acc = np.array(test_accs).min()
            val_mean_acc1 = (preds_val == y_val).mean()
            val_mean_acc2 = np.array(val_accs).mean()
            val_mean_acc3 = np.array(val_accs).min()
            if val_mean_acc3>=best_val:
                best_val = val_mean_acc3
                best_test = test_mean_acc
                worst_test = test_wors_acc
                best_val_g = val_accs
                best_test_g = test_accs

            print('Iter:', iter, 'test_mean:%.4f'%best_test, 'test_wors:%.4f'%worst_test,'val_mean:%.4f,%.4f,%.4f'%(val_mean_acc1,val_mean_acc2,val_mean_acc3))
            # print('test_acc:',best_test_g)
            print('test_acc:', test_accs)
            model.train()

    return best_test_g, best_test, best_val_g, best_val


# dname = 'cMNIST'
# dname = 'cCIFAR10'
dname = 'WB'
# dname = 'CA'

if dname == 'WB':
    args.data_dir = '/home/user/workspace/data/distri_shift/cub/data/waterbird_complete95_forest2water2/'
    args.result_path='/home/user/workspace/deep_feature_reweighting-main/results1.pkl'
    args.ckpt_path='/home/user/workspace/deep_feature_reweighting-main/results/final_checkpoint.pt'
    bias_ckp = '/home/user/workspace/deep_feature_reweighting-main/wb_results_nb/bias_checkpoint-2.pt'
    # bias_ckp = args.ckpt_path
    '''-1:74.14; 0:85.72; 1:89.56;  2:90.56; '''
    args.tune_class_weights_dfr_train = True
    args.output_dir = '/home/user/workspace/deep_feature_reweighting-main/results/'
    args.pretrained_model = True
    args.num_epochs = 20
    args.weight_decay = 1e-3
    args.batch_size = 256
    args.init_lr = 1e-5
    args.eval_freq = 1
    args.test_wb_dir = '/home/user/workspace/data/distri_shift/cub/data/waterbird_complete95_forest2water2/'
    args.augment_data = True
    args.seed = 0
    num_retrains = 3000
elif dname == 'CA':
    args.data_dir='/home/user/workspace/data/distri_shift/CelebA/data/'
    args.result_path='/home/user/workspace/deep_feature_reweighting-main/results_celeba.pkl'
    args.ckpt_path='/home/user/workspace/deep_feature_reweighting-main/results_celeba/final_checkpoint.pt'
    args.tune_class_weights_dfr_train = True
    bias_ckp = '/home/user/workspace/deep_feature_reweighting-main/results_celeba_b/bias_checkpoint-1.pt'
    # bias_ckp = args.ckpt_path
    '''-1: 40.85; 0: 89.05; 1: 89.44;  2:89.22; '''
    args.output_dir = '/home/user/workspace/deep_feature_reweighting-main/results_celeba/'
    args.pretrained_model = True
    args.num_epochs = 10
    args.weight_decay = 1e-4
    args.batch_size = 256
    args.init_lr = 1e-4
    args.eval_freq = 1
    args.test_wb_dir = '/home/user/workspace/data/distri_shift/CelebA/data/'
    args.augment_data = True
    args.seed = 0
    num_retrains = 3000
elif dname == 'cCIFAR10':
    pct = 5 #0.5 #0.5
    '''50, 63.35, 71.15, 77.45'''
    root ='/home/user/workspace/data/distri_shift/cifar10c/cifar10c/'
    args.data_dir = root + '%spct/'%pct
    args.test_wb_dir = root + 'test/'
    args.result_path='/home/user/workspace/deep_feature_reweighting-main/results_%s.pkl'%dname
    args.ckpt_path='/home/user/workspace/deep_feature_reweighting-main/results_cifar%spct/final_checkpoint.pt'%pct
    args.tune_class_weights_dfr_train = True
    bias_ckp = '/home/user/workspace/deep_feature_reweighting-main/results_cifar%spct/bias_checkpoint-1.pt'%pct
    # bias_ckp = '/home/user/workspace/deep_feature_reweighting-main/results_cifar%spct/bias_checkpoint-0.pt'%pct
    # bias_ckp = args.ckpt_path
    ''' 2:78.79; 0:77.70; 1:78.06 -1:41.71'''
    args.output_dir = '/home/user/workspace/deep_feature_reweighting-main/results_cifar%spct/'%pct
    args.pretrained_model = True
    args.num_epochs = 20
    args.weight_decay = 1e-4
    args.batch_size = 256
    args.init_lr = 1e-5
    args.eval_freq = 1
    args.augment_data = True
    num_retrains = 2000
elif dname == 'cMNIST':
    pct = 5 # 1 # 2
    root = '/home/user/workspace/data/distri_shift/cmnist/cmnist/'
    args.data_dir = root + '%spct/' % pct
    args.test_wb_dir = root + 'test/'
    args.result_path = '/home/user/workspace/deep_feature_reweighting-main/results_%s.pkl' % dname
    args.ckpt_path = '/home/user/workspace/deep_feature_reweighting-main/results_mnist%spct/final_checkpoint.pt' % pct
    args.tune_class_weights_dfr_train = True
    bias_ckp = '/home/user/workspace/deep_feature_reweighting-main/results_mnist%spct/bias_checkpoint-0.pt' % pct
    bias_ckp = args.ckpt_path
    '''-1: 55.85 ; 0: 97.56 ; 1: 97.67 ;  2: 97.81 ; '''
    args.output_dir = '/home/user/workspace/deep_feature_reweighting-main/results_mnist%spct/' % pct
    args.pretrained_model = True
    args.num_epochs = 20
    args.weight_decay = 1e-4
    args.batch_size = 256
    args.init_lr = 1e-3
    args.eval_freq = 1
    args.augment_data = True
    args.seed = 0
    num_retrains = 2000


# set_seed(args.seed)

''' data load '''
trainset, valset, testset = which_data2(dname, basedir=args.data_dir, test_dir=args.test_wb_dir)

loader_kwargs = {'batch_size': args.batch_size,
                 'num_workers': 4, 'pin_memory': False,
                 "reweight_places": None, 'drop_last':False}
train_loader = get_loader(
    trainset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)
test_loader = get_loader(
    testset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)
val_loader = get_loader(
    valset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)


'''  Model  '''
n_classes = trainset.n_classes
if dname == 'cCIFAR10':
    model = torchvision.models.resnet18(pretrained=True)
    d = model.fc.in_features
    # model.fc = torch.nn.Linear(d, n_classes)
    model.fc = torch.nn.Sequential(
        # torch.nn.Dropout(0.5),
        torch.nn.Linear(d, n_classes),
    )
    g_model = torchvision.models.resnet18(pretrained=False)
    g_model.fc = torch.nn.Linear(d, n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000], gamma=0.1)

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=args.weight_decay)

    n_groups = 10
    print('pct = ', pct, 'dname = ', dname)
elif dname == 'cMNIST':
    # model = MLP_ula(3*28*28)
    # from show_tsne import get_embed_FC as get_embed
    # from show_tsne import feat_FC as feat
    # g_model = MLP_ula(3*28*28)
    model = MNIST_CNN_fc([3,28,28])
    g_model = MNIST_CNN_fc([3,28,28])
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000], gamma=0.1)

    n_groups = 10
    print('pct = ', pct, 'dname = ', dname)
elif dname == 'CA':
    model = torchvision.models.resnet50(pretrained=True)
    d = model.fc.in_features
    # model.fc = torch.nn.Linear(d, n_classes)
    model.fc = torch.nn.Sequential(
        # torch.nn.Dropout(0.5),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(d, n_classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1)

    g_model = torchvision.models.resnet50(pretrained=False)
    g_model.fc = torch.nn.Linear(d, n_classes)
    n_groups = 4
    print('dname = ', dname)
elif dname == 'WB':
    model = torchvision.models.resnet50(pretrained=True)
    d = model.fc.in_features
    # model.fc = torch.nn.Linear(d, n_classes)
    model.fc = torch.nn.Sequential(
        # torch.nn.Dropout(0.5),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(d, n_classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000], gamma=0.1)

    g_model = torchvision.models.resnet50(pretrained=False)
    g_model.fc = torch.nn.Linear(d, n_classes)
    n_groups = 4
    print('dname = ', dname)

if dname in ['cCIFAR10', 'cMNIST','WB', 'CA']:
    model.load_state_dict(torch.load(
        args.ckpt_path
    ), strict=False)
model.cuda()
model.train()

g_model.load_state_dict(torch.load(bias_ckp))
g_model.cuda()
g_model.eval()

if dname =='cMNIST':
    all_x, all_y, gt_g, all_idx, all_place, all_embeddings = g_model.feat(train_loader)
    valall_x, valall_y, valgt_g, valall_idx, valall_place, valall_embeddings = g_model.feat(val_loader)
else:
    all_x, all_y, gt_g, all_idx, all_place, all_embeddings = feat(g_model, train_loader)
    valall_x, valall_y, valgt_g, valall_idx, valall_place, valall_embeddings = feat(g_model, val_loader)

diff = 1 - (all_place == all_y) * 1
diff_val = 1 - (valall_place == valall_y) * 1
if dname in ['cCIFAR10', 'cMNIST']:
    pseudo_g = all_place * 2 + diff
    valpseudo_g = valall_place * 2 + diff_val
elif dname=='WB':
    pseudo_g = all_y * 2 + all_place
    valpseudo_g = valall_y * 2 + valall_place
else:
    pseudo_g = all_y * 2 + 1 - all_place
    valpseudo_g = valall_y * 2 + 1 - valall_place

# n_pg = np.bincount(pseudo_g)
# tmp = n_pg.max()/n_pg.min()
# args.exp_lam = torch.exp(torch.tensor(2000.-1015)/1000.)

print("DFR on validation")
print(np.bincount(pseudo_g))
print('pred bias ratio =:', (pseudo_g==gt_g).mean())
group_paccs = [(pseudo_g==gt_g)[gt_g == g].mean()
                             for g in range(n_classes*2)]
print('pred group accuracy =:',group_paccs)

'''  if use ground truth to see the upper bound '''
if 1:
    pseudo_g = gt_g
    valpseudo_g = valgt_g
    all_place = gt_g%2

weight_sample, w_group = group_weight_specific_v4(all_place, all_y, n_classes=n_classes)

weight_sample = torch.max(weight_sample, torch.tensor(1e-4))
weight_sample = weight_sample/weight_sample.max()
print('w_group:', w_group)
tmp = w_group.max()/w_group.min()
args.exp_lam = torch.exp(torch.tensor(1020.-tmp)/1000.)
# args.exp_lam = torch.exp(torch.tensor(1020.-1020.)/1000.)
args.exp_lam = 1.

print('exp_lam = ', args.exp_lam)
print('num_retrains = ', num_retrains)
if dname == 'cCIFAR10':
    from dataset_cifar import ImageLabelFilelist as imageread
    from dataset_cifar_group_balance import ImageLabelFilelist as imageread_bl
    target_resolution = (128, 128)
    # train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=True)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            target_resolution,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=2),
        # transforms.RandomRotation(degrees=40),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    loader_kwargs = {'batch_size': 1,
                     'num_workers': 4, 'pin_memory': False,
                     "reweight_places": None, 'drop_last':False}

elif dname == 'cMNIST':
    from dataset_cmnist import ImageLabelFilelist as imageread
    from dataset_cmnist_group_balance import ImageLabelFilelist as imageread_bl
    target_resolution = (28, 28)
    # train_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            target_resolution,
            scale=(0.8, 1.1),
            ratio=(0.75, 1.3333333333333333),
            interpolation=2),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    loader_kwargs = {'batch_size': 10,
                     'num_workers': 4, 'pin_memory': False,
                     "reweight_places": None, 'drop_last':False}
else:
    from wb_data import WaterBirdsDataset as imageread
    from wb_data_Cbalance import WaterBirdsDataset as imageread_bl
    # target_resolution = (128, 128)
    target_resolution = (224, 224)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            target_resolution,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    loader_kwargs = {'batch_size': 1,
                     'num_workers': 4, 'pin_memory': False,
                     "reweight_places": None, 'drop_last':False}

if dname in ['cMNIST','cCIFAR10']:
    # trainset = imageread(args.data_dir, stage='train', transform=train_transform, group_info=gt_g)
    trainset = imageread(args.data_dir, stage='train', transform=train_transform, update_g=True)
    train_loaders = InfiniteDataLoader(dataset=trainset, weights=weight_sample, batch_size=args.batch_size,
                                            num_workers=4)
    train_minibatches_iterator = train_loaders
    trainset = imageread_bl(args.data_dir, stage='train', transform=train_transform, group_info=pseudo_g)
    train_loaders_bl = get_loader(
        trainset, train=True, reweight_groups=False, reweight_classes=False, **loader_kwargs)
    train_loaders_bl = iter(train_loaders_bl)
else:
    trainset = imageread(args.data_dir, split="train", transform=train_transform)
    train_loaders = InfiniteDataLoader(dataset=trainset, weights=weight_sample, batch_size=args.batch_size,
                                       num_workers=4)
    train_minibatches_iterator = train_loaders
    trainset = imageread_bl(args.data_dir, split='train', transform=train_transform, group_info=pseudo_g)
    train_loaders_bl = get_loader(
        trainset, train=True, reweight_groups=False, reweight_classes=False, **loader_kwargs)
    train_loaders_bl = iter(train_loaders_bl)

if dname in ['cMNIST','cCIFAR10']:
    test_accs, test_mean_acc, train_accs, train_mean_accs = dfr_on_validation_eval(
        train_minibatches_iterator, pseudo_g, test_loader, val_loader, valpseudo_g, valall_idx, num_retrains=num_retrains, dname=dname,n_groups=n_groups, weight=weight_sample, train_loaders_1=train_loaders_bl, gt_g=gt_g)
else:
    test_accs, test_mean_acc, train_accs, train_mean_accs = dfr_on_validation_eval(
        train_minibatches_iterator, pseudo_g, test_loader, val_loader, valpseudo_g, valall_idx, num_retrains=num_retrains, dname=dname,n_groups=n_groups, weight=weight_sample, train_loaders_1=train_loaders_bl, gt_g=gt_g)


''' print results '''
print('test_accs:', test_accs)
print('train_accs:', train_accs)
print('test_mean:', test_mean_acc)
print('train_mean:', train_mean_accs)

