import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

    
class Classifier(nn.Module):
    def __init__(self, n_hid, n_out, dropout = 0.5):
        super(Classifier, self).__init__()
        self.drop     = nn.Dropout(dropout)
        self.n_hids   = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(self.drop(x))
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    def __init__(self, n_hid, n_heads, dropout = 0.5):
        super(Matcher, self).__init__()
        self.left_linear    = nn.Linear(n_hid,  n_hid)
        self.right_linear   = nn.Linear(n_hid,  n_hid)
        self.merge          = nn.Linear(n_heads, 1)
        self.drop     = nn.Dropout(dropout)
        self.n_heads  = n_heads
        self.dk       = n_hid // n_heads
        self.cache      = None
    def forward(self, x, y, infer = False, pair = False):
        ty = self.drop(self.right_linear(y)).view(-1, self.n_heads, self.dk)
        if infer:
            '''
                During testing, we will consider millions or even billions of nodes as candidates (x).
                It's not possible to calculate them again for different query (y)
                Since the model is fixed, we propose to cache them, and dirrectly use the results.
            '''
            if self.cache != None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx.view(-1, self.n_heads, self.dk)
        else:
            tx = self.drop(self.left_linear(x)).view(-1, self.n_heads, self.dk)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx.transpose(0, 1), ty.transpose(0, 1).transpose(-1, -2)).transpose(0, 2)
        return torch.log_softmax(self.merge(res).squeeze(), dim=-1)
    
class BatchNorm(nn.Module):
    "Construct a batchnorm module."
    def __init__(self, features, eps=1e-6):
        super(BatchNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(0, keepdim=True)
        std  = x.std(0, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class RAGCNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.3, **kwargs):
        super(RAGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None
        
        self.interact_sws   = nn.ModuleList()
        self.interact_tws   = nn.ModuleList()
        self.transfer_sws   = nn.ModuleList()
        
        self.relation_ws   = nn.ModuleList()
        self.aggregat_ws   = nn.ModuleList()
        self.norms         = nn.ModuleList()
        
        for t in range(num_types):
            self.interact_sws.append(nn.Linear(in_dim,   out_dim))
            self.interact_tws.append(nn.Linear(in_dim,   out_dim))
            self.transfer_sws.append(nn.Linear(in_dim,   out_dim))
            self.aggregat_ws.append(nn.Linear(out_dim,  out_dim))
            self.norms.append(BatchNorm(out_dim))
            
        self.relation_ws   = nn.Parameter(torch.ones(num_types, num_relations, num_types, self.n_heads))
        self.interact_rw   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.transfer_rw   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        
        self.drop          = nn.Dropout(dropout)
        self.emb           = RelTemporalEncoding(in_dim)
        
        glorot(self.interact_rw)
        glorot(self.transfer_rw)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time, num_nodes):
        '''
            i: target; j: source
        '''
        data_size = edge_index_i.size(0)
        atts, vals = [], []
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_val     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_id in range(self.num_types):
            sb = (node_type_j == int(source_id))
            interact_sw = self.interact_sws[source_id]
            transfer_sw = self.transfer_sws[source_id] 
            for target_id in range(self.num_types):
                tb = (node_type_i == int(target_id)) & sb
                interact_tw = self.interact_tws[target_id]
                for relation_id in range(self.num_relations):
                    idx = ((edge_type == int(relation_id)) * tb).detach()
                    if idx.sum() == 0:
                        continue
                    _node_inp_i = node_inp_i[idx]
                    _node_inp_j = self.emb(node_inp_j[idx], edge_time[idx])
                    _int_i = interact_tw(_node_inp_i).view(-1, self.n_heads, self.d_k)
                    _int_j = interact_sw(_node_inp_j).view(-1, self.n_heads, self.d_k)
                    _int_s = torch.bmm(_int_j.transpose(1,0), self.interact_rw[relation_id]).transpose(1,0)
                    tmp = (_int_s * _int_i).sum(dim=-1) * self.relation_ws[target_id][relation_id][source_id] / self.sqrt_dk
                    res_att[idx] = tmp
                    _tra_j = transfer_sw(_node_inp_j).view(-1, self.n_heads, self.d_k)
                    res_val[idx] = torch.bmm(_tra_j.transpose(1,0), self.transfer_rw[relation_id]).transpose(1,0)
                    
        self.att = softmax(res_att, edge_index_i, data_size)
        res = res_val * self.att.view(-1, self.n_heads, 1)
        del res_att, res_val
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
           x = BN[node_type](W[node_type] * GNN(x))
        '''
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for t_id in range(self.num_types):
            aggregat_w = self.aggregat_ws[t_id]
            norm = self.norms[t_id]
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = norm(F.relu(aggregat_w(aggr_out[idx])))
        out = self.drop(res)
        del res
        return out

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.3):
        super(RelTemporalEncoding, self).__init__()
        self.drop = nn.Dropout(dropout)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid * 2, 2.)) / n_hid / 2)
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid * 2, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.drop(self.emb(t)))
    
    
    
    
    