import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl
from torch.nn import init
from dgl.utils import expand_as_pair
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.readout import sum_nodes

# gasa model
class gasa_classifier(torch.nn.Module):
    def __init__(self, dropout, num_heads, hidden_dim1, hidden_dim3, hidden_dim2, n_tasks=2, in_dim=1):
        super(gasa_classifier, self).__init__()
        self.gnn = global_attention(in_dim, hidden_dim3 * num_heads, num_heads, edge_feat_size=11, dim=hidden_dim1)       
        self.gat1 = GATConv(hidden_dim1, hidden_dim2, num_heads,
                          negative_slope=0.2, bias=True) 
        self.gat2 = GATConv(hidden_dim2 * num_heads, hidden_dim2, 1,
                          negative_slope=0.2, bias=True)
        self.gat3 = GATConv(hidden_dim2, hidden_dim2, 1,
                          negative_slope=0.2, bias=True)
        self.readout = WeightedSumAndMax(hidden_dim2)
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim3),
            nn.Linear(hidden_dim3, n_tasks))
        self.dropout = nn.Dropout(p=dropout)
       
    
    def forward(self, g, get_node_weight=True):
        edge_feats = g.edata['he'] 
        h = g.in_degrees().view(-1, 1).float()
        h = self.gnn(g, h, edge_feats)
        h = torch.flatten(self.dropout(F.elu(self.gat1(g, h))), 1)
        h = torch.flatten(self.dropout(F.elu(self.gat2(g, h))), 1)
        h = torch.mean(self.dropout(F.elu(self.gat3(g, h))), 1)
        g_feats, node_weights = self.readout(g, h, get_node_weight)
        hg = self.predict(g_feats)
        return hg


    # for SHAP explaination
    # def forward(self, g, h, mask, nei, aver, get_node_weight=True):
    #     edge_feats = g.edata['he'] 
    #     if mask == False:
    #         h = self.gnn(g, h, edge_feats)
    #         h = torch.flatten(self.dropout(F.elu(self.gat1(g, h))), 1)
    #         h = torch.flatten(self.dropout(F.elu(self.gat2(g, h))), 1)
    #         h = torch.mean(self.dropout(F.elu(self.gat3(g, h))), 1)
    #         g_feats, node_weights = self.readout(g, h, get_node_weight)
    #         hg = self.predict(g_feats)
    #         f = torch.softmax(hg, 1)
    #     elif mask == True:
    #         h = self.gnn(g, h, edge_feats)
    #        #shield atom feature duiring massage passing
    #         for val in nei:
    #             h[val, :] = aver
    #         h = torch.flatten(self.dropout(F.elu(self.gat1(g, h))), 1)
    #         for val in nei:
    #             h[val, :] = aver 
    #         h = torch.flatten(self.dropout(F.elu(self.gat2(g, h))), 1)
    #         for val in nei:
    #             h[val, :] = aver 
    #         h = torch.mean(self.dropout(F.elu(self.gat3(g, h))), 1) 
    #         for val in nei:
    #             h[val, :] = aver
    #         g_feats, node_weights = self.readout(g, h, get_node_weight)
    #         hg = self.predict(g_feats)
    #         f = torch.softmax(hg, 1)
    #     return f


class global_attention(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 dim,
                 edge_feat_size,
                 negative_slope=0.2,
                 feat_drop=0,
                 residual=False,
                 allow_zero_in_degree=False,
                 bias=True):
        super(global_attention, self).__init__()
        self._num_heads = num_heads
        self.dim = dim
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_feat_size = edge_feat_size
        self.full = nn.Linear(out_feats * num_heads, out_feats)
        self.linears1 = nn.Linear(out_feats + edge_feat_size, out_feats)
        self.linears2 = nn.Linear(out_feats * 2, dim)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
                    #print('There are 0-in-degree nodes in the graph')
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = edge_softmax(graph, e)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            new = torch.flatten(F.leaky_relu(rst), 1)
            ft = self.full(new)
            graph.ndata['h'] = ft
            graph.edata['he'] = edge_feat
            graph.apply_edges(lambda edges: {'he1': torch.cat([edges.src['h'], edges.data['he']], dim=1)})
            graph.edata['he1'] = torch.tanh(self.linears1(graph.edata['he1']))
            graph.ndata['hv_new'] = ft
            graph.apply_edges(lambda egdes: {
                'he2': torch.cat([egdes.dst['hv_new'], graph.edata['he1']], dim=1)})
            graph.update_all(fn.copy_e('he2', 'm'), fn.mean('m', 'a'))
            hf = graph.ndata.pop('a')
            global_g = torch.tanh(self.linears2(hf))
            return global_g

class WeightAndSum(nn.Module):
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1)
        )

    def forward(self, g, feats):
        with g.local_scope():
            g.ndata['h'] = feats
            atom_weights = self.atom_weighting(g.ndata['h'])
            #print(atom_weight)
            g.ndata['w'] = torch.nn.Sigmoid()(self.atom_weighting(g.ndata['h']))
            h_g_sum = sum_nodes(g, 'h', 'w')
        return h_g_sum, atom_weights

class WeightedSumAndMax(nn.Module):

    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats, get_node_weight=False):
        h_g_sum = self.weight_and_sum(bg, feats)[0]
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        atom_weight = self.weight_and_sum(bg, feats)[1]
        if get_node_weight:
            return h_g, atom_weight
        else:
            return h_g