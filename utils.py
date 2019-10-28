import json
import math, copy, time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import pandas as pd
import scipy.sparse as sp

import torch
import math
import seaborn as sb
import dill
from functools import partial
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm

class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        )))))
        self.times = {}
    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]
    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]
    def sqrt_norm(self, x, l):
        return x / np.sqrt(l) - np.sqrt(l) / 2
            
    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    def get_types(self):
        return list(self.node_feature.keys())
    def propagate_feature(self):
        '''
            Since only paper have w2v embedding, we simply propagate its
            feature to other nodes by averaging neighborhoods.
            Then, we construct the Dataframe for each node type.
        '''
        d = pd.DataFrame(self.node_bacward['paper'])
        self.node_feature = {'paper': d}
        cv = np.array(list(d['emb']))
        for _type in self.node_bacward:
            if _type != 'paper':
                d = pd.DataFrame(self.node_bacward[_type])
                i = []
                for _rel in self.edge_list[_type]['paper']:
                    for j in self.edge_list[_type]['paper'][_rel]:
                        for t in self.edge_list[_type]['paper'][_rel][j]:
                            i += [[j, t]]
                if len(i) == 0:
                    continue
                i = np.array(i).T
                v = np.ones(i.shape[1])
                m = normalize(sp.coo_matrix((v, i), \
                    shape=(len(self.node_bacward[_type]), len(self.node_bacward['paper']))))
                out = m.dot(cv)
                d['emb'] = list(out)
                self.node_feature[_type] = d
        '''
            Affiliation is not directly linked with Paper, so we average the author embedding.
        '''
        if 'author' in self.node_bacward and 'affiliation' in self.node_bacward:
            cv = np.array(list(self.node_feature['author']['emb']))
            d = pd.DataFrame(self.node_bacward['affiliation'])
            i = []
            for _rel in self.edge_list['affiliation']['author']:
                for j in self.edge_list['affiliation']['author'][_rel]:
                    for t in self.edge_list['affiliation']['author'][_rel][j]:
                        i += [[j, t]]
            i = np.array(i).T
            v = np.ones(i.shape[1])
            m = normalize(sp.coo_matrix((v, i), \
                shape=(len(self.node_bacward['affiliation']), len(self.node_bacward['author']))))
            out = m.dot(cv)
            d['emb'] = list(out)
            self.node_feature['affiliation'] = d


def sample_subgraph(graph, time_range, sampled_depth = 2, sampled_number = 8, inp = None):
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacancy matrix.
    '''
    layer_data  = defaultdict( #target_type
                        lambda: {} # {target_id + time}
                    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id + source_time
                                        lambda: 0. #sampled_score
                            ))
    new_layer_adj  = defaultdict( #target_type
                                    lambda: defaultdict(  #source_type
                                        lambda: defaultdict(  #relation_type
                                            lambda: [] #[target_id, source_id]
                                )))
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self':
                    continue
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    k = encode(source_id, source_time)
                    '''
                        If the node's time is out of range or already being sampled, skip
                        Otherwise, accumulate the normalized degree.
                    '''
                    if source_time not in time_range or k in layer_data[source_type]:
                        continue
                    budget[source_type][k] += 1. / len(sampled_ids)
    '''
        The encode and decode function is used to index each node
        by its node_id and time together. So that a same node with
        different timestamps can exist in the sampled graph.
    '''
    def decode(s):
        idx = s.find('-')
        return np.array([s[:idx], s[idx+1:]], dtype=float)
    def encode(i, t):
        return '%s-%s' % (i, t)

    '''
        If inp == None: we sample some paper as initial nodes;
        else:           we are dealing with a specific supervised task (e.g. author disambiguation),
                        where some node pairs are given as output.
    '''
    
    if inp == None:
        _time = np.random.choice(list(time_range.keys()))
        res = graph.node_feature['paper'][graph.node_feature['paper']['time'] == _time]
        sampn = min(len(res), sampled_number)
        rand_paper_ids  = np.random.choice(list(res.index), sampn, replace = False)
        '''
            First adding the sampled nodes then updating budget.
        '''
        for _id in rand_paper_ids:
            layer_data['paper'][encode(_id, _time)] = len(layer_data['paper'])
        for _id in rand_paper_ids:
            add_budget(graph.edge_list['paper'], _id, _time, layer_data, budget)
    else:
        '''
            First adding the sampled nodes then updating budget.
        '''
        for _type in inp:
            for _id, _time in inp[_type]:
                layer_data[_type][encode(_id, _time)] = len(layer_data[_type])
        for _type in inp:
            te = graph.edge_list[_type]
            for _id, _time in inp[_type]:
                add_budget(te, _id, _time, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            else:
                '''
                    Sample based on accumulated degree
                '''
                score = np.array(list(budget[source_type].values())) ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = len(layer_data[source_type])
            for k in sampled_keys:
                source_id, source_time = decode(k)
                add_budget(te, int(source_id), int(source_time), layer_data, budget)
                budget[source_type].pop(k)    
    '''
        Prepare feature, time and adjacency matrix for the sampled graph
    '''
    feature = {}
    times   = {}
    indxs   = {}
    for _type in layer_data:
        idxs  = np.array([decode(key) for key in layer_data[_type]])
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature['field'].loc[idxs[:,0], 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs[:,0], 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs[:,0], 'citation'])).reshape(-1, 1) + 0.01)), axis=1)
        
        times[_type]   = idxs[:,1]
        indxs[_type]   = idxs[:,0]
    edge_list = defaultdict( #target_type
                        lambda: defaultdict(  #source_type
                            lambda: defaultdict(  #relation_type
                                lambda: [] # [target_id, source_id] 
                                    )))
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    '''
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_key in layer_data[target_type]:
                    target_ser = layer_data[target_type][target_key]
                    tesrt = tesr[decode(target_key)[0]]
                    for source_key in layer_data[source_type]:
                        source_ser = layer_data[source_type][source_key]
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        '''
                        if decode(source_key)[0] in tesrt:
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
    return feature, times, edge_list, indxs

def to_torch(feature, time, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_feature = []
    node_type    = []
    node_time    = []
    edge_index   = []
    edge_type    = []
    edge_time    = []
    
    node_num = 0
    types = graph.get_types()
    for t in graph.get_types():
        type_id = len(node_dict)
        node_dict[t] = [node_num, type_id]
        node_num     += len(feature[t])
    if 'fake_paper' in feature:
        node_dict['fake_paper'] = [node_num, node_dict['paper'][1]]
        node_num     += len(feature['fake_paper'])
        types += ['fake_paper']
    for t in types:
        node_feature += list(feature[t])
        node_time    += list(time[t])
        type_id = node_dict[t][1]
        node_type    += [type_id for _ in range(len(feature[t]))]
        
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)
    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ti, si in edge_list[target_type][source_type][relation_type]:
                    sid, tid = si + node_dict[source_type][0], ti + node_dict[target_type][0]
                    edge_index += [[sid, tid]]
                    edge_type  += [edge_dict[relation_type]]   
                    '''
                        Our time ranges from 1900 - 2020, largest span is 120.
                    '''
                    edge_time  += [node_time[tid] - node_time[sid] + 120]
    node_feature = torch.FloatTensor(node_feature)
    node_type    = torch.LongTensor(node_type)
    edge_time    = torch.LongTensor(edge_time)
    edge_index   = torch.LongTensor(edge_index).t()
    edge_type    = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]