import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from typing import Optional
from torch_geometric.typing import OptTensor
from torch.nn import Linear
from torch_geometric.nn.inits import glorot, zeros
import networkx as nx
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch_geometric.utils import num_nodes
from torch_scatter import scatter_add,scatter_softmax
from torch_scatter.scatter import scatter_mean, scatter_sum


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

    def forward(self, x, edge_index, edge_weight,batch):
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
        out=self.propagate(edge_index, x=x, norm=norm)

        return out, None

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(GATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight,batch):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            num_nodes = x_r.size(0) if x_r is not None else num_nodes
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out,None

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class ChebConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K,decoupled, normalization='sym',
                 bias=True, **kwargs):
        super(ChebConv, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        
        self.decoupled=decoupled
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.weight = Parameter(torch.Tensor(K*in_channels, out_channels))
        self.node_att = nn.Linear(K*in_channels,1)
        self.K=K

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

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

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = [Tx_0]
        # out = torch.matmul(Tx_0, self.weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.K> 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            # out = out + torch.matmul(Tx_1, self.weight[1])
            out.append(Tx_1)

        for k in range(2, self.K):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            # out = out + torch.matmul(Tx_2, self.weight[k])
            out.append(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        out=torch.stack(out).permute(1,0,2).contiguous()

        node_score=self.node_att(out.reshape(-1,self.K*self.in_channels))
        out=torch.matmul(out.reshape(-1,self.K*self.in_channels),self.weight)

        if self.decoupled:
            out=out/torch.norm(out,dim=1,keepdim=True)
            out=out*node_score

        return out,node_score

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)

class MixhopConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K,decoupled, normalization='sym',
                 bias=True, **kwargs):
        super(MixhopConv, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        
        self.decoupled=decoupled
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.hidden=out_channels//K
        self.weight=Parameter(torch.Tensor(K*in_channels,out_channels))

        self.K=K

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        row, col = edge_index[0], edge_index[1]
        edge_weight = torch.ones(row.shape[0],device=row.device)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

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

        num_nodes=x.size(self.node_dim)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = [torch.mm(Tx_0,self.weight[:self.in_channels,:self.hidden])]
        # propagate_type: (x: Tensor, norm: Tensor)
        if self.K> 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            # out = out + torch.matmul(Tx_1, self.weight[1])
            out.append(torch.mm(Tx_1,self.weight[self.in_channels:2*self.in_channels,self.hidden:2*self.hidden]))

        for k in range(2, self.K):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            # out = out + torch.matmul(Tx_2, self.weight[k])
            if k<self.K-1:
                out.append(torch.mm(Tx_2,self.weight[k*self.in_channels:(k+1)*self.in_channels,k*self.hidden:(k+1)*self.hidden]))
            else:
                out.append(torch.mm(Tx_2,self.weight[k*self.in_channels:,k*self.hidden:]))
            Tx_1 = Tx_2

        out=torch.cat(out,dim=1)
        return out,None

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)

class LiCheb(MessagePassing):
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True,decoupled=True, **kwargs):
        super(LiCheb, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.K=K
        self.decoupled=decoupled
        self.fc = Linear(in_channels,out_channels)
        self.att_w = Parameter(torch.Tensor(K, out_channels))
        self.att_bias = Parameter(torch.Tensor(out_channels))
        self.node_att = nn.Linear(K*out_channels,1)
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

        ## for importance score
        node_score=self.node_att(out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels))
        
        out=torch.sum(self.att_w.view(self.K,1,self.out_channels)*out,dim=0)+self.att_bias.view(1,self.out_channels)
        if self.decoupled:
            out=out/torch.norm(out,dim=1,keepdim=True)
        out=out*node_score

        return out,node_score

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)

# class LiCheb(MessagePassing):
#     def __init__(self, in_channels, out_channels, K, normalization='sym',
#                  bias=True,decoupled=True, **kwargs):
#         super(LiCheb, self).__init__(aggr='add', **kwargs)

#         assert K > 0
#         assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.normalization = normalization
#         self.K=K
#         self.decoupled=decoupled
#         self.fc = Linear(in_channels,out_channels)
#         self.att_w = Parameter(torch.Tensor(K, out_channels))
#         self.att_bias = Parameter(torch.Tensor(out_channels))
#         self.node_att = nn.Linear(K*out_channels,1)
#         self.edge_att = nn.Linear(K*out_channels*2,1)
#         self.reset_parameters()

#     def reset_parameters(self):
#         # glorot(self.weight)
#         # zeros(self.bias)
#         glorot(self.att_w)
#         zeros(self.att_bias)

#     def __norm__(self, edge_index, num_nodes: Optional[int],
#                  edge_weight: OptTensor, normalization: Optional[str],
#                  lambda_max, dtype: Optional[int] = None,
#                  batch: OptTensor = None):

#         edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

#         edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
#                                                 normalization, dtype,
#                                                 num_nodes)

#         if batch is not None and lambda_max.numel() > 1:
#             lambda_max = lambda_max[batch[edge_index[0]]]

#         edge_weight = (2.0 * edge_weight) / lambda_max
#         edge_weight.masked_fill_(edge_weight == float('inf'), 0)

#         edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
#                                                  fill_value=-1.,
#                                                  num_nodes=num_nodes)
#         assert edge_weight is not None

#         return edge_index, edge_weight

    # def forward(self, x, edge_index, edge_weight: OptTensor = None,
    #             batch: OptTensor = None, lambda_max: OptTensor = None):
    #     """"""
    #     if self.normalization != 'sym' and lambda_max is None:
    #         raise ValueError('You need to pass `lambda_max` to `forward() in`'
    #                          'case the normalization is non-symmetric.')

    #     if lambda_max is None:
    #         lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
    #     if not isinstance(lambda_max, torch.Tensor):
    #         lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
    #                                   device=x.device)
    #     assert lambda_max is not None

    #     edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
    #                                      edge_weight, self.normalization,
    #                                      lambda_max, dtype=x.dtype,
    #                                      batch=batch)

    #     x=self.fc(x)
    #     Tx_0 = x
    #     Tx_1 = x  # Dummy.
    #     out = [Tx_0]

    #     # propagate_type: (x: Tensor, norm: Tensor)
    #     if self.K > 1:
    #         Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
    #         out.append(Tx_1)

    #     for k in range(2, self.K):
    #         Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
    #         Tx_2 = 2. * Tx_2 - Tx_0
    #         out.append(Tx_2)
    #         Tx_0, Tx_1 = Tx_1, Tx_2

    #     out=torch.stack(out)

    #     ## for importance score
    #     out2=out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels)
    #     node_score=self.node_att(out2)
    #     # row,col=edge_index
    #     # edge_score = self.edge_att(torch.cat([out2[row],out2[col]],dim=1))
        
    #     out=torch.sum(self.att_w.view(self.K,1,self.out_channels)*out,dim=0)+self.att_bias.view(1,self.out_channels)
    #     if self.decoupled:
    #         out=out/torch.norm(out,dim=1,keepdim=True)
    #     out=out*node_score

    #     return out,node_score

    # def forward(self, x, edge_index, edge_weight: OptTensor = None,
    #             batch: OptTensor = None, lambda_max: OptTensor = None):
    #     """"""
    #     if self.normalization != 'sym' and lambda_max is None:
    #         raise ValueError('You need to pass `lambda_max` to `forward() in`'
    #                          'case the normalization is non-symmetric.')

    #     if lambda_max is None:
    #         lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
    #     if not isinstance(lambda_max, torch.Tensor):
    #         lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
    #                                   device=x.device)
    #     assert lambda_max is not None

    #     edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
    #                                      edge_weight, self.normalization,
    #                                      lambda_max, dtype=x.dtype,
    #                                      batch=batch)

    #     x=self.fc(x)
    #     Tx_0 = x
    #     Tx_1 = x  # Dummy.
    #     out = [Tx_0]

    #     # propagate_type: (x: Tensor, norm: Tensor)
    #     if self.K > 1:
    #         Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
    #         out.append(Tx_1)

    #     for k in range(2, self.K):
    #         Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
    #         Tx_2 = 2. * Tx_2 - Tx_0
    #         out.append(Tx_2)
    #         Tx_0, Tx_1 = Tx_1, Tx_2

    #     out=torch.stack(out)

    #     ## for importance score
    #     neighbor_info=out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels)
    #     out=torch.sum(self.att_w.view(self.K,1,self.out_channels)*out,dim=0)+self.att_bias.view(1,self.out_channels)
    #     if self.decoupled:
    #         out=out/torch.norm(out,dim=1,keepdim=True)

    #     return out,neighbor_info

    # def message(self, x_j, norm):
    #     return norm.view(-1, 1) * x_j

    # def __repr__(self):
    #     return '{}({}, {}, K={}, normalization={})'.format(
    #         self.__class__.__name__, self.in_channels, self.out_channels,
    #         self.weight.size(0), self.normalization)

class LiMixhop(MessagePassing):
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True,decoupled=True, **kwargs):
        super(LiMixhop, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.K=K
        self.decoupled=decoupled
        self.fc = Linear(in_channels,out_channels)
        self.att_w = Parameter(torch.Tensor(K, out_channels))
        self.att_bias = Parameter(torch.Tensor(out_channels))
        self.node_att = nn.Linear(K*out_channels,1)
        self.edge_att = nn.Linear(K*out_channels*2,1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_w)
        zeros(self.att_bias)

    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        row, col = edge_index[0], edge_index[1]
        edge_weight = torch.ones(row.shape[0],device=row.device)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

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

        num_nodes=x.size(self.node_dim)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)

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
            out.append(Tx_2)
            Tx_1 = Tx_2

        out=torch.stack(out)

        ## for importance score
        neighbor_info=out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels)
        out=torch.sum(self.att_w.view(self.K,1,self.out_channels)*out,dim=0)+self.att_bias.view(1,self.out_channels)
        if self.decoupled:
            out=out/torch.norm(out,dim=1,keepdim=True)

        return out,neighbor_info

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)




from torch_sparse import spspmm, coalesce
def k_hop_edges(edge_index,num_nodes,K):
    # edge_index, _ = remove_self_loops(edge_index, None)
    n = num_nodes
    edge_index,_= coalesce(edge_index, None, n, n)
    value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

    edges=[]
    new_edge_index = edge_index.clone()
    new_edge_value = value.clone()
    useful_edge_index, edge_weight = add_self_loops(new_edge_index, None,fill_value=-1.,num_nodes=num_nodes)
    edges.append(useful_edge_index)
    for i in range(K-1):
        new_edge_index, new_edge_value = spspmm(new_edge_index, new_edge_value, edge_index, value, n, n, n)
        new_edge_index, new_edge_value = coalesce(new_edge_index, new_edge_value, n, n)
        useful_edge_index, edge_weight = add_self_loops(new_edge_index, None,fill_value=-1.,num_nodes=num_nodes)
        edges.append(useful_edge_index)

    # edge_index, _ = add_self_loops(edge_index, None,fill_value=-1.,num_nodes=num_nodes)
    return edges


class MeanConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True,decoupled=True, **kwargs):
        super(MeanConv, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.K=K
        self.decoupled=decoupled
        self.fc = Linear(in_channels,out_channels)
        self.att_w = Parameter(torch.Tensor(K, out_channels))
        self.att_bias = Parameter(torch.Tensor(out_channels))
        self.node_att = nn.Linear(K*out_channels,1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_w)
        zeros(self.att_bias)

    def forward(self, x, edge_index, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        num_nodes=x.shape[0]
        x=self.fc(x)
        Tx_0 = x
        out = [Tx_0]

        # propagate_type: (x: Tensor, norm: Tensor)
        khop_edge_index=k_hop_edges(edge_index,num_nodes,self.K)
        for k in range(1, self.K):
            edge_index=khop_edge_index[k-1]
            row,col=edge_index[0],edge_index[1]
            Tx_2=scatter_mean(x[row],col,dim=0)
            out.append(Tx_2)

        out=torch.stack(out)

        ## for importance score
        node_score=self.node_att(out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels))
        
        out=torch.sum(self.att_w.view(self.K,1,self.out_channels)*out,dim=0)+self.att_bias.view(1,self.out_channels)
        return out,node_score

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)



class AttConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True,decoupled=True, **kwargs):
        super(AttConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.K=K
        self.decoupled=decoupled
        self.fc = Linear(in_channels,out_channels)
        self.key = Linear(in_channels,out_channels)
        self.query = Linear(in_channels,out_channels)

        self.att_w = Parameter(torch.Tensor(K, out_channels))
        self.att_bias = Parameter(torch.Tensor(out_channels))
        self.node_att = nn.Linear(K*out_channels,1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_w)
        zeros(self.att_bias)

    def forward(self, x, edge_index, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        num_nodes=x.shape[0]
        query=self.query(x)
        key=self.key(x)
        x=self.fc(x)
        

        Tx_0 = x
        out = [Tx_0]

        # propagate_type: (x: Tensor, norm: Tensor)
        khop_edge_index=k_hop_edges(edge_index,num_nodes,self.K)
        for k in range(1, self.K):
            edge_index=khop_edge_index[k-1]
            row,col=edge_index[0],edge_index[1]
            score=torch.sum(query[row]*key[col],dim=1)
            normed_score=scatter_softmax(score,col)
            mask=normed_score>1e-5
            Tx_2=scatter_sum(x[row[mask]]*normed_score[mask].view(-1,1),col[mask],dim=0)
            out.append(Tx_2)

        out=torch.stack(out)

        ## for importance score
        node_score=self.node_att(out.permute(1,0,2).contiguous().view(-1,self.K*self.out_channels))
        
        out=torch.sum(self.att_w.view(self.K,1,self.out_channels)*out,dim=0)+self.att_bias.view(1,self.out_channels)
        if self.decoupled:
            out=out/torch.norm(out,dim=1,keepdim=True)
            out=out*node_score

        return out,node_score

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)