import torch
from torch import random
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from layers_conv import GCN,GATConv,ChebConv,LiCheb,MeanConv,AttConv,MixhopConv,LiMixhop
from layers_pool import LookHopsPool,ASAPooling,EdgePooling,SAGPooling,TopKPooling,NoPool
import collections
import torch.nn as nn
import networkx as nx
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_layers=args.num_layers
        self.K=args.K
        
        self.num_classes = args.num_classes
        self.pooling_nodes = args.pooling_nodes
        self.pooling_edges=args.pooling_edges
        
        #############################  conv method: GCN/ChebConv/GAT/LiCheb
        if args.conv=='GCN':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',GCN(self.num_features,self.nhid))]+
                [('conv{}'.format(i),GCN(self.nhid,self.nhid)) for i in range(2,self.num_layers+1)]
            ))
        
        if args.conv=='ChebConv':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',ChebConv(self.num_features,self.nhid,K=args.K,decoupled=args.decoupled))]+
                [('conv{}'.format(i),ChebConv(self.nhid,self.nhid,K=args.K,decoupled=args.decoupled)) for i in range(2,self.num_layers+1)]
            ))
        
        if args.conv=='GAT':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',GATConv(self.num_features,self.nhid,heads=1))]+
                [('conv{}'.format(i),GATConv(self.nhid,self.nhid,heads=1)) for i in range(2,self.num_layers+1)]
            ))

        if args.conv=='LiCheb':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',LiCheb(self.num_features,self.nhid,K=args.K,decoupled=args.decoupled))]+
                [('conv{}'.format(i),LiCheb(self.nhid,self.nhid,K=args.K,decoupled=args.decoupled)) for i in range(2,self.num_layers+1)]
            ))
        
        if args.conv=='LiMixhop':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',LiMixhop(self.num_features,self.nhid,K=args.K,decoupled=args.decoupled))]+
                [('conv{}'.format(i),LiMixhop(self.nhid,self.nhid,K=args.K,decoupled=args.decoupled)) for i in range(2,self.num_layers+1)]
            ))
        
        if args.conv=='MeanConv':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',MeanConv(self.num_features,self.nhid,K=args.K,decoupled=args.decoupled))]+
                [('conv{}'.format(i),MeanConv(self.nhid,self.nhid,K=args.K,decoupled=args.decoupled)) for i in range(2,self.num_layers+1)]
            ))

        if args.conv=='Mixhop':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',MixhopConv(self.num_features,self.nhid,K=args.K,decoupled=args.decoupled))]+
                [('conv{}'.format(i),MixhopConv(self.nhid,self.nhid,K=args.K,decoupled=args.decoupled)) for i in range(2,self.num_layers+1)]
            ))
        
        if args.conv=='AttConv':
            self.convs = nn.Sequential(collections.OrderedDict(
                [('conv1',AttConv(self.num_features,self.nhid,K=args.K,decoupled=args.decoupled))]+
                [('conv{}'.format(i),AttConv(self.nhid,self.nhid,K=args.K,decoupled=args.decoupled)) for i in range(2,self.num_layers+1)]
            ))

        #############################  pool method: GCN/ChebConv/GAT/LiCheb
        if args.pool=='NoPool':
            self.pool=NoPool()
        
        if args.pool=='TopkPool':
            self.pool=TopKPooling(self.nhid,self.pooling_nodes)
        
        if args.pool=='SAGPool':
            self.pool=SAGPooling(self.nhid,self.pooling_nodes)
        
        if args.pool=='EdgePool':
            self.pool=EdgePooling(self.nhid)
        
        if args.pool=='ASAPPool':
            self.pool=ASAPooling(self.nhid,self.pooling_nodes)

        if args.pool=='LookHops':
            self.pool = LookHopsPool(self.K,self.pooling_nodes,self.pooling_edges)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        self.edges=[]

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = None

        x_out=0
        for idx in range(self.num_layers-1):
            self.edges.append(edge_index)
            x,x_score=self.convs[idx](x, edge_index, edge_weight, batch)
            x = F.relu(x)  # x: [69403, 89]  --> [69403, 128]
            x, edge_index, edge_weight, batch = self.pool(x,x_score, edge_index, edge_weight, batch) # x: [34769, 128]
            x_ = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # x1: [256, 256]
            x_out+=F.relu(x_)
        
        x,x_score=self.convs[self.num_layers-1](x, edge_index, edge_weight, batch)
        x = F.relu(x)  # x: [69403, 89]  --> [69403, 128]
        x_ = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x_out+=F.relu(x_)

        x = F.relu(self.lin1(x_out))
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x

    # def forward(self, data):
    #     x, edge_index, batch = data.x, data.edge_index, data.batch
    #     edge_weight = None

    #     G=nx.from_edgelist(data.edge_index.cpu().numpy().T)
    #     Dis=dict(nx.all_pairs_shortest_path_length(G,cutoff=3))
    #     edge,D=[],[]
    #     for i in Dis.keys():
    #         for j in Dis[i].keys():
    #             edge.append([i,j])
    #             D.append(Dis[i][j])
    #     edge=np.array(edge).T
    #     D=np.array(D)
    #     mask=~(edge[0]==edge[1])
    #     edge_pool=edge[:,mask]
    #     D_pool=torch.tensor(D[mask],device=x.device).float()

    #     x_out=0
    #     for idx in range(self.num_layers-1):
    #         if idx==0:
    #             x,neighbor_info=self.convs[idx](x, edge_index, edge_weight, batch)
    #             edge_index=edge_pool
    #             edge_dis=D_pool
    #         else:
    #             x,neighbor_info=self.convs[idx](x, edge_index, edge_weight, batch)

    #         x = F.relu(x)  # x: [69403, 89]  --> [69403, 128]
    #         x, edge_index, edge_weight,edge_dis, batch = self.pool(x,neighbor_info, edge_index, edge_dis, batch) # x: [34769, 128]
    #         x_ = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # x1: [256, 256]
    #         x_out+=F.relu(x_)
        
    #     x,x_score=self.convs[self.num_layers-1](x, edge_index, edge_weight, batch)
    #     x = F.relu(x)  # x: [69403, 89]  --> [69403, 128]
    #     x_ = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
    #     x_out+=F.relu(x_)

    #     x = F.relu(self.lin1(x_out))
    #     x = F.relu(self.lin2(x))
    #     x = F.log_softmax(self.lin3(x), dim=-1)
    #     return x