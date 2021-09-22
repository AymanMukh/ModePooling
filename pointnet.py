# Ayman: Added Histogram (HS) and RANSAC (RN) pooling operations

# MODELNET40 CLASSIFICATION WITH DECLARATIVE ROBUST POOLING NODES
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>

# Modified from PyTorch PointNet code:
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/31deedb10b85ec30178df57a6389b2f326f7c970

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import scipy.io
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
# sys.path.append("../../../")
# import ddn.pytorch.robustpool as robustpool

class STN3d(nn.Module):
    def __init__(self, robust_type='', alpha=1.0):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.robust_type = robust_type
        self.alpha = alpha

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Adjust pooling layer
        x = batch_hist(x, bins=73, min=-10, max=10)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = batch_hist(x, bins=73, min=-10, max=10)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

L=1024

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, input_transform=False, feature_transform=False, robust_type='', alpha=1.0, semseg=False):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(robust_type) if not semseg else STNkd(k=9)
        self.conv1 = torch.nn.Conv1d(3, 64, 1) if not semseg else torch.nn.Conv1d(9, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, L, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(L)
        self.global_feat = global_feat
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.robust_type = robust_type
        self.alpha = alpha

    def forward(self, x):

        n_pts = x.size()[2]
        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Adjust pooling layer
        alpha_tensor = torch.tensor([self.alpha], dtype=x.dtype, device=x.device, requires_grad=False)
        if self.robust_type == 'HS':
            x = batch_hist(x, bins=70, min=-10, max=10)
        elif self.robust_type == 'RN':
            x = ransac(x, it=.5, d=.143)
        # elif self.robust_type == 'Q':
        #     x = robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.Quadratic, alpha_tensor)
        # elif self.robust_type == 'PH':
        #     x = robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.PseudoHuber, alpha_tensor)
        # elif self.robust_type == 'H':
        #     x = robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.Huber, alpha_tensor)
        # elif self.robust_type == 'W':
        #     x = robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.Welsch, alpha_tensor)
        # elif self.robust_type == 'TQ':
        #     x= robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.TruncatedQuadratic, alpha_tensor)

        else:
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, input_transform=False, feature_transform=False, robust_type='', alpha=1.0, model='pointnet'):
        super(PointNetCls, self).__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        if model=='pointnet':
            self.feat = PointNetEncoder(global_feat=True, input_transform=input_transform, feature_transform=feature_transform, robust_type=robust_type, alpha=alpha)
        else:
            self.feat = DGCNN(global_feat=True, input_transform=input_transform,
                                        feature_transform=feature_transform, robust_type=robust_type, alpha=alpha)

        self.fc1 = nn.Linear(L, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()



    def forward(self, x):


        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans_feat

class DGCNN(nn.Module):
    def __init__(self, global_feat=True, input_transform=False, feature_transform=False, robust_type='', alpha=1.0,
                 semseg=False):
        super(DGCNN, self).__init__()
        self.stn = STN3d(robust_type) if not semseg else STNkd(k=9)
        self.conv1 = torch.nn.Conv2d(6, 64, 1) if not semseg else torch.nn.Conv1d(9, 64, 1)
        self.conv2 = torch.nn.Conv2d(64*2, 128, 1)
        self.conv3 = torch.nn.Conv2d(128*2, 1024, 1)
        self.conv4 = torch.nn.Conv1d(64, 1024, 1)   # change 64 to 1024 for all conv layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.robust_type = robust_type
        self.alpha = alpha

    def forward(self, x):
        b,f,n_pts = x.size()  # 10 ,3 , 1024

        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        idx = knn(x, k=20)
        x = get_graph_feature_d(x, idx=idx, k=20, d=.25)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.max(dim=-1, keepdim=False)[0]

        trans_feat = None
        pointfeat = x

        # x = get_graph_feature_d(x,  idx=idx, k=20, d=2)
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = x.max(dim=-1, keepdim=False)[0]
        # x = get_graph_feature_d(x,  idx=idx, k=20, d=2)
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = x.max(dim=-1, keepdim=False)[0]
        x = self.bn4(self.conv4(x))

        # Adjust pooling layer
        alpha_tensor = torch.tensor([self.alpha], dtype=x.dtype, device=x.device, requires_grad=False)
        if self.robust_type == 'HS':
            x = batch_hist(x, bins=70, min=-10, max=10)
        elif self.robust_type == 'RN':
            x = ransac(x, it=.5, d=.143)
        # elif self.robust_type == 'Q':
        #     x = robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.Quadratic, alpha_tensor)
        # elif self.robust_type == 'PH':
        #     x = robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.PseudoHuber, alpha_tensor)
        # elif self.robust_type == 'H':
        #     x = robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.Huber, alpha_tensor)
        # elif self.robust_type == 'W':
        #     x = robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.Welsch, alpha_tensor)
        # elif self.robust_type == 'TQ':
        #     x= robustpool.RobustGlobalPool2dFn.apply(x.unsqueeze(-1), robustpool.TruncatedQuadratic, alpha_tensor)
        else:
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, cat_num=16,part_num=50):
        super(PointNetDenseCls, self).__init__()
        self.cat_num = cat_num
        self.part_num = part_num
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        # classification network
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, cat_num)
        self.dropout = nn.Dropout(p=0.3)
        self.bnc1 = nn.BatchNorm1d(256)
        self.bnc2 = nn.BatchNorm1d(256)
        # segmentation network
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud,label):
        batchsize,_ , n_pts = point_cloud.size()
        # point_cloud_transformed
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        point_cloud_transformed = torch.bmm(point_cloud, trans)
        point_cloud_transformed = point_cloud_transformed.transpose(2, 1)
        # MLP
        out1 = F.relu(self.bn1(self.conv1(point_cloud_transformed)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        # net_transformed
        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)
        # MLP
        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)
        # classification network
        net = F.relu(self.bnc1(self.fc1(out_max)))
        net = F.relu(self.bnc2(self.dropout(self.fc2(net))))
        net = self.fc3(net) # [B,16]
        # segmentation network
        out_max = torch.cat([out_max,label],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, n_pts)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net2 = F.relu(self.bns1(self.convs1(concat)))
        net2 = F.relu(self.bns2(self.convs2(net2)))
        net2 = F.relu(self.bns3(self.convs3(net2)))
        net2 = self.convs4(net2)
        net2 = net2.transpose(2, 1).contiguous()
        net2 = F.log_softmax(net2.view(-1, self.part_num), dim=-1)
        net2 = net2.view(batchsize, n_pts, self.part_num) # [B, N 50]

        return net, net2, trans_feat

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature

def ransac(x, it=.5, d=.143):
    batch_size = x.size(0)
    num_points = x.size(2)
    num_features = x.size(1)
    if it != 1:
        idx = np.arange(num_points )
        np.random.shuffle(idx)
        num_samples= int(it*num_points)
        x = x[...,  idx]
        y1 = x[..., 0:num_samples].unsqueeze(-1)
    else:
        y1 = x.unsqueeze(-1)
    # x is batch_size, num_points, num features
    # distance is x-x
    y = x.unsqueeze(-1).transpose(3, 2)

    # perform looping , number of loops = 80 , num_points / 80 = 25
  #  dist = torch.cuda.IntTensor(x.size(0), x.size(1), num_samples).fill_(0)
    # for i in range(80):
    #     y2 = torch.abs(y1 - y[...,(0+i*25):(25+i*25)])
    #     cond = torch.le(y2, d)
    #     dist1 = torch.sum(cond, -1)
    #     dist = dist.add(dist1)

    # pairwise_distance
    y = torch.abs(y1 - y)
    cond = torch.le(y, d)
    dist = torch.sum(cond, -1)

    y = torch.max(dist, -1)[1].unsqueeze(-1)
    x = x.gather(-1, y).squeeze(-1)

    return x

def get_edge_feature_d(x, idx=None, k=20, d=.2):
    x=x.view(x.size(0), 1)
    if idx is None:
        # x is batch_size, num_points, num features
        # distance is x-x
        pairwise_distance = abs(x - x.transpose(1, 0))
        # dist= pairwise_distance.topk(k=k, dim=-1)[0]

    cond = torch.le(pairwise_distance, d)
    dist = torch.sum(cond, 1)
    feature = torch.where(cond, feature, torch.zeros_like(feature))

    dist = torch.sum(-dist,1)
    dist = torch.min(dist, 0)[0]
    # num_points,num_dims  = x.size()
    #
    # # x = x.transpose(1, 0).contiguous()
    # feature = x[idx, :]
    # feature = feature.view(num_points, k, num_dims)
    # x = x.view(num_points, 1, num_dims).repeat( 1, k, 1)
    # dist = abs(feature - x)  # tf.norm(point_cloud_cd,axis=3,keepdims=True)
    # dist = torch.sum(dist, 1)
    # dist = torch.min(dist, 0)[0]

  #  cond = torch.le(dist, d)  # cond = tf.less_equal(point_cloud_ncd, d)
  #  #cond = cond.view(batch_size, num_points, k, 1).repeat(1, 1, 1, num_dims)  # cond = tf.tile(cond, [1, 1, 1,  num_dims ])
  #  feature = torch.where(cond, feature, torch.zeros_like(
   #     feature))  # point_cloud_neighbors = tf.where(cond, point_cloud_neighbors, tf.zeros(point_cloud_cd.shape))
    return dist

def batch_hist(x, bins=70, min=-10, max=10):

    # h = torch.cuda.FloatTensor(x.size(0), x.size(1)).fill_(0)
    # for jj in range(x.size(0)):
    #     for jK in range(x.size(1)):
    #         outputs.append(torch.histc(x[jj, jK, :], bins=70, min=-10, max=10))
    #         _, h[jj, jK] = torch.max(torch.histc(x[jj, jK, :], bins=70, min=-10, max=10), 0)
    #
    # xx=x
    batch_size = x.size(0)
    num_points = x.size(2)
    num_features = x.size(1)
  
    b = torch.linspace(0, (max - min) * num_features * batch_size, steps=num_features * batch_size+1,device=torch.device('cuda'))

    b = b[0:-1].view(batch_size* num_features, -1).repeat(1,num_points).view(batch_size* num_features*num_points)
    x = b + x.view(-1)
    output1 = torch.histc(x, bins=bins * num_features * batch_size, min=min, max=(max-min) * num_features * batch_size+min)
    x = output1.view(batch_size, num_features, -1)
    # x = torch.where(x > 0, torch.linspace(min, max, steps=bins, device=torch.device('cuda')).view(1, 1, bins).repeat(batch_size, num_features, 1), torch.zeros_like(x))  # max pooling
    # x = torch.max(x, 2, keepdim=True)[0].view(-1, num_features).float()
    x = torch.max(x, 2, keepdim=True)[1].view(-1, num_features).float()
    return x

def normalize(x):
    x_normed = torch.div(x, x.max(-1, keepdim=True)[0])
    return x_normed

def get_graph_feature_d(x, idx=None, k=20, d=.2):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    dist = abs(feature - x)  # tf.norm(point_cloud_cd,axis=3,keepdims=True)
    cond = torch.le(dist, d)  # cond = tf.less_equal(point_cloud_ncd, d)
    # cond = cond.view(batch_size, num_points, k, 1).repeat(1, 1, 1, num_dims)  # cond = tf.tile(cond, [1, 1, 1,  num_dims ])
   # feature = torch.where(cond, feature, torch.zeros_like(feature))  # point_cloud_neighbors = tf.where(cond, point_cloud_neighbors, tf.zeros(point_cloud_cd.shape))
    feature = torch.where(cond, feature, x)
    return feature

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I).cpu(), dim=(1, 2)))
        loss = loss.cuda()
        # loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss

class PointNetLoss(torch.nn.Module):
    def __init__(self, weight=1,mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.weight = weight

    def forward(self, labels_pred, label, seg_pred,seg, trans_feat):
        seg_loss = F.nll_loss(seg_pred, seg)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        label_loss = F.nll_loss(labels_pred, label)

        loss = self.weight * seg_loss + (1-self.weight) * label_loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss, seg_loss, label_loss

class PointNetSeg(nn.Module):
    def __init__(self,num_class,feature_transform=False, semseg = False):
        super(PointNetSeg, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False,feature_transform=feature_transform, semseg = semseg)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat



if __name__ == '__main__':
    point = torch.randn(8,3,1024)
    label = torch.randn(8,16)
    model = PointNetDenseCls()
    net, net2, trans_feat = model(point,label)
    print('net',net.shape)
    print('net2',net2.shape)
    print('trans_feat',trans_feat.shape)
