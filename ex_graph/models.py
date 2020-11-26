import torch
from torch import random
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv,ChebConv
import scipy
from layers import GCN, HGPSLPool,LiCheb,LeCheb,AttPool
import collections
import torch.nn as nn

# class Model(torch.nn.Module):
#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.args = args
#         self.num_features = args.num_features
#         self.nhid = args.nhid
#         self.num_classes = args.num_classes
#         self.pooling_ratio = args.pooling_ratio
#         self.dropout_ratio = args.dropout_ratio
#         self.sample = args.sample_neighbor
#         self.sparse = args.sparse_attention
#         self.sl = args.structure_learning
#         self.lamb = args.lamb
#         self.edge_ratio=args.edge_ratio

#         if args.conv=='GCN':
#             self.conv1 = GCNConv(self.num_features, self.nhid)
#             self.conv2 = GCN(self.nhid, self.nhid)
#             self.conv3 = GCN(self.nhid, self.nhid)
        
#         if args.conv=='ChebConv':
#             self.conv1 = ChebConv(self.num_features,self.nhid,K=args.K)
#             self.conv2 = ChebConv(self.nhid,self.nhid,K=args.K)
#             self.conv3 = ChebConv(self.nhid,self.nhid,K=args.K)
        
#         if args.conv=='LiCheb':
#             self.conv1 = LiCheb(self.num_features,self.nhid,K=args.K)
#             self.conv2 = LiCheb(self.nhid,self.nhid,K=args.K)
#             self.conv3 = LiCheb(self.nhid,self.nhid,K=args.K)
#             self.conv4 = LiCheb(self.nhid,self.nhid,K=args.K)
        
#         if args.conv=='LeCheb':
#             self.conv1 = LeCheb(self.num_features,self.nhid,K=args.K)
#             self.conv2 = LeCheb(self.nhid,self.nhid,K=args.K)
#             self.conv3 = LeCheb(self.nhid,self.nhid,K=args.K)
        
#         if args.conv=='Mix':
#             self.conv1 = LiCheb(self.num_features,self.nhid,K=args.K)
#             self.conv2 = LiCheb(self.nhid,self.nhid,K=args.K)
#             self.conv3 = LeCheb(self.nhid,self.nhid,K=args.K)

#         if args.pool=='HGPSL':
#             self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
#             self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        
#         if args.pool=='MAtt':
#             self.pool1 = AttPool(self.pooling_ratio)
#             self.pool2 = AttPool(self.pooling_ratio)
            
#         # self.pool1 = InfoPool(self.nhid,self.pooling_ratio,self.edge_ratio)
#         # self.pool2 = InfoPool(self.nhid,self.pooling_ratio,self.edge_ratio)

#         self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
#         self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
#         self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

#         self.weight = Parameter(torch.rand(4))

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         edge_attr = None

#         x,x_score1=self.conv1(x, edge_index, edge_attr, batch)
#         x = F.relu(x)  # x: [69403, 89]  --> [69403, 128]
#         x, edge_index, edge_attr, batch = self.pool1(x,x_score1, edge_index, edge_attr, batch) # x: [34769, 128]
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # x1: [256, 256]

#         edge_attr=None
#         x,x_score2=self.conv2(x, edge_index, edge_attr, batch)
#         x = F.relu(x)
#         x, edge_index, edge_attr, batch = self.pool2(x,x_score2, edge_index, edge_attr, batch)
#         x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # x2: [256, 256]

#         edge_attr=None
#         x,x_score3=self.conv3(x, edge_index, edge_attr, batch)
#         x = F.relu(x)
#         x, edge_index, edge_attr, batch = self.pool2(x,x_score3, edge_index, edge_attr, batch)
#         x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # x3: [256, 256]

#         edge_attr=None
#         x,x_score4=self.conv4(x, edge_index, edge_attr, batch)
#         x = F.relu(x)
#         x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # x3: [256, 256]

#         x = F.relu(x1) + F.relu(x2) + F.relu(x3) + F.relu(x4)

#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)
#         x = F.relu(self.lin2(x))
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)
#         x = F.log_softmax(self.lin3(x), dim=-1)
#         return x,x_score1,x_score2


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.K=args.K
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb
        self.edge_ratio=args.edge_ratio
        self.num_layers=args.num_layers

        if args.conv=='LiCheb':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',LiCheb(self.num_features,self.nhid,K=args.K))]+
                [('conv{}'.format(i),LiCheb(self.nhid,self.nhid,K=args.K)) for i in range(2,self.num_layers+1)]
            ))
        
        if args.pool=='MAtt':
            self.pool = AttPool(self.K,self.pooling_ratio,self.edge_ratio)
            # self.pool2 = AttPool(self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x_out=0
        for idx in range(self.num_layers-1):
            x,x_score=self.convs[idx](x, edge_index, edge_attr, batch)
            x = F.relu(x)  # x: [69403, 89]  --> [69403, 128]
            x, edge_index, edge_attr, batch = self.pool(x,x_score, edge_index, edge_attr, batch) # x: [34769, 128]
            x_ = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # x1: [256, 256]
            x_out+=F.relu(x_)
        
        x,x_score=self.convs[self.num_layers-1](x, edge_index, edge_attr, batch)
        x = F.relu(x)  # x: [69403, 89]  --> [69403, 128]
        x_ = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x_out+=F.relu(x_)

        x = F.relu(self.lin1(x_out))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x
