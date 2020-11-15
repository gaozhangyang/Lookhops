from math import degrees
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import threshold
from torch_scatter.scatter import scatter_mean
from torch_sparse.tensor import to
from sparse_softmax import Sparsemax
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter_add,scatter_softmax,scatter_max,scatter_std
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from typing import Optional
from torch_geometric.typing import OptTensor
from torch.nn import Linear
from torch_geometric.nn.inits import glorot, zeros
import networkx as nx


class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n)
        value.fill_(0)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super(GCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class SparseActivate(nn.Module):
    def __init__(self):
        super(SparseActivate,self).__init__()
        self.threshold=Parameter(torch.tensor([-0.99]))
        self.activation=nn.ReLU()
        self.f=nn.Sigmoid()
    
    def forward(self,x,batch):
        # print(self.f(self.threshold))
        tmp, argmax = scatter_max(x, batch)
        max_value=torch.zeros(x.shape[0],device=x.device)
        max_value[argmax]=1
        max_mask=max_value.bool()

        value=self.activation(x-self.threshold)
        max_value=torch.max(torch.tensor([torch.max(value),torch.ones(1,device=x.device)]))
        # print(max_value)
        merged_value=value*(~max_mask)+max_value*(max_mask)
        
        # mean=scatter_mean(x,batch)[batch]
        # std=scatter_std(x,batch)[batch]

        # normed_value=(x-mean)/(std+1e-10)
        # cutted_value=self.activation(normed_value-self.f(self.threshold))
        # mask_cut=cutted_value>0

        # merged_value=((cutted_value*std+mean)*mask_cut)*(~max_mask) +max_value
        return merged_value



# def sparseFunction(x, s,batch, activation=torch.relu,f=torch.sigmoid):
#     mean=torch.mean(x)
#     var=torch.var(x)
#     normed_value=(x-mean)/var
#     mask=activation(normed_value-f(s))
#     return (mask*var+mean)*(mask>0).float()

class LiCheb(MessagePassing):
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True, **kwargs):
        super(LiCheb, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.K=K
        self.fc = Linear(in_channels,out_channels)
        self.att_w = Parameter(torch.Tensor(K, out_channels))
        self.att_bias = Parameter(torch.Tensor(out_channels))
        self.node_att = nn.Linear(K*out_channels,1)
        # self.threshold = Parameter(torch.tensor([-0.9]))
        self.SA = SparseActivate()
        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        # zeros(self.bias)
        glorot(self.att_w)
        zeros(self.att_bias)

    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)

        x=self.fc(x)
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = [Tx_0]

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.K > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out.append(Tx_1)

        for k in range(2, self.K):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out.append(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        out=torch.stack(out)
        ## for learnable threshold
        # node_score=self.node_att(out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels))
        # node_score=self.SA(node_score.view(-1),batch).view(-1,1)

        ## for importance score
        node_score=self.node_att(out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels))
        
        out=torch.sum(self.att_w.view(self.K,1,self.out_channels)*out,dim=0)+self.att_bias.view(1,self.out_channels)
        out=out/torch.norm(out,dim=1,keepdim=True)
        out=out*node_score

        return out,node_score

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)


class LeCheb(MessagePassing):
    def __init__(self, in_channels, out_channels, K, normalization='sym', **kwargs):
        super(LeCheb, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.K=K
        self.fc = Linear(in_channels,out_channels)
        self.attN_w = Parameter(torch.Tensor(K, 3*out_channels))

        self.attK_w = Parameter(torch.Tensor(K, out_channels))
        self.attK_bias = Parameter(torch.Tensor(out_channels))
        self.node_att = nn.Linear(K*out_channels,1)
        # self.threshold = Parameter(torch.tensor([-0.9]))
        self.SA = SparseActivate()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.attN_w)
        glorot(self.attK_w)
        zeros(self.attK_bias)
    
    def attention(self,x,edge_index,k):
        # calculate center and corresponding neighbor
        row,col=edge_index[0],edge_index[1]
        attention=F.relu( torch.sum(self.attN_w[k-2].view(1,-1)*torch.cat([x[row],x[col],x[row]*x[col]],dim=1),dim=1) )
        attention=scatter_softmax(attention,row)
        return attention


    def forward(self, x, edge_index, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        x=self.fc(x)
        out = [x]

        n = x.shape[0]
        edge_weight=torch.ones(edge_index.shape[1]).to(edge_index.device)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,fill_value=-1.,num_nodes=x.size(self.node_dim))

        if self.K>1:
            new_edge_index, new_value = spspmm(edge_index, edge_weight, edge_index, edge_weight, n, n, n,coalesced=True)
            attention=self.attention(x,new_edge_index,1)
            x_new = self.propagate(new_edge_index, x=x, norm=attention, size=None)
            out.append(x_new)

            

        for k in range(2, self.K):
            new_edge_index,new_value = spspmm(edge_index, edge_weight, new_edge_index, new_value, n, n, n,coalesced=True)
            attention=self.attention(x,new_edge_index,k)
            x_new = self.propagate(new_edge_index, x=x, norm=attention, size=None)
            out.append(x_new)

        out=torch.stack(out)

        ## for learnable threshold
        # node_score=self.node_att(out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels))
        # node_score=self.SA(node_score.view(-1),batch).view(-1,1)

        ## for importance score
        node_score=self.node_att(out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels))

        out=torch.sum(self.attK_w.view(self.K,1,self.out_channels)*out,dim=0)+self.attK_bias.view(1,self.out_channels)
        out=out/torch.norm(out,dim=1,keepdim=True)
        out=out*node_score

        return out,node_score

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class AttPool(torch.nn.Module):
    def __init__(self, ratio=0.8):
        super(AttPool, self).__init__()
        self.ratio = ratio

        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, x_score, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # Graph Pooling
        perm = topk(x_score.view(-1), self.ratio, batch)
        # perm=torch.where(x_score.view(-1)>0)[0] # for learnable pooling
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=x_score.size(0))
        return x, induced_edge_index, induced_edge_attr, batch


class InformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(InformationScore, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_weight):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.shape[0])
        
        row,col = edge_index[0], edge_index[1]
        diff_edge = torch.norm(x[row]-x[col],p=2,dim=1)
        diff_x = torch.ones(x.shape[0],1,device=x.device)
        diff_x = self.propagate(edge_index, x=diff_x, norm=diff_edge) # degrees
        return diff_x.view(-1)/(deg.view(-1)+1e-10), diff_edge

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


    
class HGPSLPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2):
        super(HGPSLPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.negative_slop = negative_slop
        self.lamb = lamb

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_information_score = self.calc_information_score(x, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)

        # Graph Pooling
        original_x = x
        perm = topk(score, self.ratio, batch)
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        # Discard structure learning layer, directly return
        if self.sl is False:
            return x, induced_edge_index, induced_edge_attr, batch

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each pair of nodes is time consuming.
            # To accelerate this process, we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.
            k_hop = 3
            if edge_attr is None:
                edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)

            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            for _ in range(k_hop - 1):
                hop_data = self.neighbor_augment(hop_data) # TODO
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, perm, num_nodes=score.size(0)) # TODO

            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))
            # row, col = new_edge_index
            # weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            # weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb
            # adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            # adj[row, col] = weights
            # new_edge_index, weights = dense_to_sparse(adj)
            # row, col = new_edge_index
            # if self.sparse:
            #     new_edge_attr = self.sparse_attention(weights, row)
            # else:
            #     new_edge_attr = softmax(weights, row, x.size(0))
            # # filter out zero weight edges
            # adj[row, col] = new_edge_attr
            # new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # # release gpu memory
            # del adj
            # torch.cuda.empty_cache()


        return x, new_edge_index, new_edge_attr, batch

