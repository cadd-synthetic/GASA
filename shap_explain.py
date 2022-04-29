import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem, rdDepictor
import dgl
import matplotlib.cm as cm
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import display, SVG
from classifier import gasa_classifier
import torch
from data import predict_collate
import pandas as pd
from torch.utils.data import DataLoader
from data import pred_data
import os
import numpy as np
from itertools import combinations
import scipy.special
import torch.nn.functional as F
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
import random
from torch_geometric.data import Data
from copy import deepcopy
from sklearn.metrics import r2_score
import torch_geometric
import heapq
import shap
from gasa_utils import generate_graph
# we use SHAP method for explain our model
class GraphSVX():
    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.gpu = gpu
        self.neighbours = None  #  nodes considered
        self.F = None  # number of features considered
        self.M = None  # number of features and nodes considered
        self.base_values = []
        self.model.eval()

    def explain_graphs(self,
                       hops=2,
                       num_samples=200,
                       info=True,
                       multiclass=False,
                       fullempty=None,
                       S=3,
                       args_hv='compute_pred',
                       args_feat='Expectation',
                       args_coal='Smarter',
                       args_g='WLS',
                       regu=0,
                       vizu=False):
        """ Explains prediction for a graph classification task - GraphSVX method
        Args:
            node_indexes (list, optional): indexes of the nodes of interest. Defaults to [0].
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph 
                                                    around node_index. Defaults to 2.#考虑2跳邻居对节点v的影响
            num_samples (int, optional): number of samples we want to form GraphSVX's new dataset. 
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings. 
                                                    And include vizualisation. Defaults to True.
            multiclass (bool, optional): extension - consider predicted class only or all classes
            fullempty (bool, optional): enforce high weight for full and empty coalitions
            S (int, optional): maximum size of coalitions that are favoured in mask generation phase 
            args_hv (str, optional): strategy used to convert simplified input z to original
                                                    input space z'
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z
            args_g (str, optional): method used to train model g on (z, f(z'))
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features
            vizu (bool, optional): creates vizualisation or not
        Returns:
                [tensors]: shapley values for features/neighbours that influence node v's pred
                        and base value
        """
        # --- Explain several nodes iteratively ---
        phi_list = []
        # Compute true prediction for original instance via explained GNN model
        ndata = deepcopy(self.data.ndata['hv'])
        edata = deepcopy(self.data.edata['he'])
        with torch.no_grad():
            true_conf, true_pred = self.model(self.data, h=self.data.in_degrees().view(-1, 1).float(), mask=False, nei=None, aver=None)[0,:].max(dim=0)
        print(true_conf, true_pred)
        # Remove node v index from neighbours and store their number in D
        node_indices = self.data.nodes().numpy().tolist()
        D = []
        self.neighbours = []
        for node_index in node_indices: 
            neighbours, _, _, edge_mask = torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                num_hops=hops,
                                                edge_index=(self.data.adj()._indices()).long())
         
            neighbours = neighbours[neighbours != node_index]
            self.neighbours.append(list(neighbours.numpy()))
            number = neighbours.shape[0]
            D.append(number)
        D = len(D)
        self.F = 0 
        self.M = self.F+D
        # Def range of endcases considered
        args_K = S   
        # --- MASK GENERATOR ---
        z_, weights = self.mask_generation(num_samples, args_coal, args_K, D, info, regu)
        if fullempty:
            weights[(weights == 1000).nonzero()] = 0

        # --- GRAPH GENERATOR --- 
        # Create dataset (z, f(GEN(z'))), stored as (z_, fz)
        # Retrieve z' from z and x_v, then compute f(z')
        fz = self.graph_classification(
                num_samples, D, z_, args_K, args_feat, true_pred)
        # --- EXPLANATION GENERATOR --- 
        # Train Surrogate Weighted Linear Regression - learns shapley values
        phi, base_value, r2 = eval('self.' + args_g)(z_, weights, fz,
                                                    multiclass, info)
        phi_list.append(phi)
        self.base_values.append(base_value)
        self.data.ndata['hv'] = ndata
        self.data.edata['he'] = edata
        return phi_list, self.base_values, r2
       

    def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
     
        def maybe_num_nodes(index, num_nodes=None):
            return index.max().item() + 1 if num_nodes is None else num_nodes
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        assert flow in ['source_to_target', 'target_to_source']
        if flow == 'target_to_source':
            row, col = edge_index
        else:
            col, row = edge_index
        node_mask = row.new_empty(num_nodes, dtype=torch.bool)
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device).flatten()
        else:
            node_idx = node_idx.to(row.device)
        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]
        node_mask.fill_(False)
        node_mask[subset] = True
        edge_mask = node_mask[row] & node_mask[col]

        edge_index = edge_index[:, edge_mask]

        if relabel_nodes:
            node_idx = row.new_full((num_nodes, ), -1)
            node_idx[subset] = torch.arange(subset.size(0), device=row.device)
            edge_index = node_idx[edge_index]

        return subset, edge_index, inv, edge_mask

    def feature_selection(self, node_index, args_feat):
        #Only consider relevant features in explanations
        discarded_feat_idx = []
        if args_feat == 'All':
            # Select all features 
            self.F = self.data.x[node_index, :].shape[0]
            feat_idx = torch.unsqueeze(
                torch.arange(self.data.num_nodes), 1)
        elif args_feat == 'Null':
            # Select features whose value is non-null
            feat_idx = self.data.x[node_index, :].nonzero()
            self.F = feat_idx.size()[0]
        else:
            # Select features whose value is different from dataset mean value
            std = self.data.ndata['hv'].std(axis=0)
            mean = self.data.ndata['hv'].mean(axis=0)
            mean_subgraph = self.data.ndata['hv'][node_index, :]
            mean_subgraph = torch.where(mean_subgraph >= mean - 0.25*std, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
            mean_subgraph = torch.where(mean_subgraph <= mean + 0.25*std, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
            feat_idx = (mean_subgraph == 100).nonzero()
            discarded_feat_idx = (mean_subgraph != 100).nonzero()
            self.F = feat_idx.shape[0]

        return feat_idx, discarded_feat_idx    
    

    def mask_generation(self, num_samples, args_coal, args_K, D, info, regu):
        """ Applies selected mask generator strategy 
        Args:
            num_samples (int): number of samples for GraphSVX 
            args_coal (str): mask generator strategy 
            args_K (int): size param for indirect effect 
            D (int): number of nodes considered after selection
            info (bool): print information or not 
            regu (int): balances importance granted to nodes and features
        Returns:
            [tensor] (num_samples, M): dataset of samples/coalitions z' 
            [tensor] (num_samples): vector of kernel weights corresponding to samples 
        """
        if args_coal == 'SmarterSeparate' or args_coal == 'NewSmarterSeparate':
            weights = torch.zeros(num_samples, dtype=torch.float64)
            if self.F==0 or D==0:
                num = int(num_samples * self.F/self.M)
            elif regu != None:
                num = int(num_samples * regu)
                #num = int( num_samples * ( self.F/self.M + ((regu - 0.5)/0.5)  * (self.F/self.M) ) )    
            else: 
                num = int(0.5* num_samples/2 + 0.5 * num_samples * self.F/self.M)
            # Features only
            z_bis = eval('self.' + args_coal)(num, args_K, 1)  
            z_bis = z_bis[torch.randperm(z_bis.size()[0])]
            s = (z_bis != 0).sum(dim=1)
            weights[:num] = self.shapley_kernel(s, self.F)
            z_ = torch.zeros(num_samples, self.M)
            z_[:num, :self.F] = z_bis
            # Node only
            z_bis = eval('self.' + args_coal)(
                num_samples-num, args_K, 0)  
            z_bis = z_bis[torch.randperm(z_bis.size()[0])]
            s = (z_bis != 0).sum(dim=1)
            weights[num:] = self.shapley_kernel(s, D)
            z_[num:, :] = torch.ones(num_samples-num, self.M)
            z_[num:, self.F:] = z_bis
        else:
            # If we choose to sample all possible coalitions
            if args_coal == 'All':
                num_samples = min(10000, 2**self.M)

            # Coalitions: sample num_samples binary vectors of dimension M 
            z_ = eval('self.' + args_coal)(num_samples, args_K, regu)

            # Shuffle them 
            z_ = z_[torch.randperm(z_.size()[0])]

            # Compute |z| for each sample z: number of non-zero entries
            s = (z_ != 0).sum(dim=1)

            # GraphSVX Kernel: define weights associated with each sample 
            weights = self.shapley_kernel(s, self.M)
        return z_, weights

    def Smarter(self, num_samples, args_K, *unused):
        """ Smart Mask generator
        Nodes and features are considered together but separately
        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling 
        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        i = 2
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 9*(num_samples - i)//10
                while i < samp and k <= args_K:
                    # Sample coalitions of k1 neighbours or k1 features without repet and order.
                    L = list(combinations(range(self.F), k)) + \
                        list(combinations(range(self.F, self.M), k))
                    random.shuffle(L)
                    L = L[:samp+1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(
                                num_samples-i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(
                                num_samples-i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                return z_       
        return z_

    def shapley_kernel(self, s, M):
        """ Computes a weight for each newly created sample 
        Args:
            s (tensor): contains dimension of z for all instances
                (number of features + neighbours included)
            M (tensor): total number of features/nodes in dataset
        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shapley_kernel = []

        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                # Enforce high weight on full/empty coalitions
                shapley_kernel.append(1000)
            elif scipy.special.binom(M, a) == float('+inf'):
                # Treat specific case - impossible computation
                shapley_kernel.append(1/ (M**2))
            else:
                shapley_kernel.append(
                    (M-1)/(scipy.special.binom(M, a)*a*(M-a)))

        shapley_kernel = np.array(shapley_kernel)
        shapley_kernel = np.where(shapley_kernel<1.0e-40, 1.0e-40,shapley_kernel)
        return torch.tensor(shapley_kernel)
    
    def graph_classification(self, num_samples, D, z_, args_K, args_feat, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Graph Classification task
        Args:
            Variables are defined exactly as defined in explainer function
            Note that adjacency matrices are dense (square) matrices (unlike node classification)
        Returns:
            (tensor): f(z') - probability of belonging to each target classes, for all samples z'
            Dimension (N * C) where N is num_samples and C num_classses.
        """
     
        excluded_nei = {}
        for i in range(num_samples):
            # Excluded nodes' indexes 
            #print(self.neighbours[0])
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    for m in self.neighbours[j]:
                        nodes_id.append(m)
            excluded_nei[i] = list(set(nodes_id))
            #Dico with key = num_sample id, value = excluded neighbour index
        # Init 
        fz = torch.zeros(num_samples)
        f = (self.data.adj()._indices()).long()
        # if args_feat == 'Null':
        #     av_feat_values = torch.zeros(self.data[0].shape[1])
        # else: 
        
        # Create new matrix A and X - for each sample ≈ reform z' from z
        ndata = deepcopy(self.data.ndata['hv']) 
        edata = deepcopy(self.data.edata['he'])
        av_feat_values = ndata.mean(dim=0).mean(dim=0)
        for (key, ex_nei) in excluded_nei.items():
            X = self.data.ndata['hv']
            Y = self.data.edata['he']
            DJ = self.data.in_degrees().view(-1, 1).float()
            if len(ex_nei) == 0:
                self.data.ndata['hv'] = ndata
                self.data.edata['he'] = edata
            elif len(ex_nei) != 0:
                for nei in ex_nei:
                    X[nei, :] = av_feat_values
                    DJ[nei] = 0
                    index = np.argwhere(f[0].numpy() == nei)
                    index2 = np.argwhere(f[1].numpy() == nei)
                    true_nei = np.concatenate((index, index2),axis=0)
                    Y[true_nei, :] = 0
                    self.data.ndata['hv'] = X
                    self.data.edata['he'] = Y
       
            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(X.unsqueeze(0).cuda(), A.unsqueeze(0).cuda()).exp()
            else:
                with torch.no_grad():
                    proba = self.model(self.data, h=DJ, mask=True, nei=ex_nei, aver=av_feat_values)
                  
            # Compute prediction
           
            fz[key] = proba[0][true_pred.item()]

        return fz

    
    def WLS(self, z_, weights, fz, multiclass, info):
        """ Weighted Least Squares Method
            Estimates shapley values via explanation model
        Args:
            z_ (tensor): binary vector representing the new instance
            weights ([type]): shapley kernel weights for z
            fz ([type]): prediction f(z') where z' is a new instance - formed from z and x
        Returns:
            [tensor]: estimated coefficients of our weighted linear regression - on (z, f(z'))
            Dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)
        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible 
            if info:
                print('WLS: Matrix not invertible')
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(10**(-5) * np.random.randn(tmp.shape[1])))

        phi = np.dot(tmp, np.dot(
            np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))
 
        # Test accuracy
        y_pred = z_.detach().numpy() @ phi
        r2 = r2_score(fz, y_pred)
        print('r2: ', r2_score(fz, y_pred))
        print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1], r2



def draw(smiles, graph, atom_weights, p):
    bg = dgl.batch([graph])
    min_value = min(atom_weights)
    max_value = max(atom_weights)
    atom_weights = (atom_weights - min_value) / (max_value - min_value)
    mol = Chem.MolFromSmiles(smiles)
    norm = cm.colors.Normalize(vmin=0, vmax=1.00)
    print(atom_weights)
    cmap = cm.get_cmap('OrRd')
    plt.figure(figsize=(6,4))
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i]) for i in range(bg.number_of_nodes())}
    #plt.colorbar(plt_colors, ticks=np.arange(0.2, 1.2, step=0.2))
    rdDepictor.Compute2DCoords(mol)
    dr = rdMolDraw2D.MolDraw2DSVG(300, 300)
    do = rdMolDraw2D.MolDrawOptions()
    do.bondLineWidth = 4
    do.fixedBondLength = 30
    do.highlightRadius = 4
    dr.SetFontSize(0.7)
    dr.drawOptions().addAtomIndices = True
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    dr.DrawMolecule(mol)
    dr.FinishDrawing()
    svg = dr.GetDrawingText()
    svg = svg.replace('svg:', '')
    return (smiles, atom_weights, svg)

#load molecules

smiles = 'CC1=CC2C([NH3+])CC2([NH+](C)C)C1'
graph = generate_graph(smiles)[0]
model = gasa_classifier(dropout=0.1, num_heads=6, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32)
checkpoint = torch.load('./gasa.pth',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
for i in range(len(graph)):
    phi= GraphSVX(data=graph[0], model=model)
    m, base, r2 = phi.explain_graphs()
    smile, aw, svg = draw(smiles, graph, list(m[0]), i)
    print(m, base, r2)
    display(SVG(svg))

