import torch.nn.functional as functional
import torch.nn as nn
import torch
import pdb
from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Parameter

import random
import numpy as np
from model.ens_nets.GCN import GCNConv as ens_GCNConv
from model.ens_nets.GAT import GATConv as ens_GATConv
from model.ens_nets.SAGE import SAGEConv as ens_SAGEConv



class GNN(nn.Module):
    def __init__(self,model, in_channels,out_channels,aggr_methode,**kwargs):
        
        super(GNN, self).__init__()
        global GCNConv
        from torch_geometric.nn import GCNConv,GATConv,GINConv,SAGEConv
        # In case ens is active, we need the different model
        self.kwargs = kwargs
        special_keys={}
        if kwargs['ens']:
            GCNConv = ens_GCNConv
            GATConv = ens_GATConv
            SAGEConv = ens_SAGEConv
            special_keys['cached'] = False
            special_keys['normalize'] = True
        
        self.bn0 = nn.BatchNorm1d(in_channels)
        self.bn1 = nn.BatchNorm1d(kwargs['hidden_channels'])
        
        if model == "custom_GCN":
            

            self.lin1 = SAGEConv(in_channels, kwargs['hidden_channels'])
            self.conv1 = GCNConv(kwargs['hidden_channels'], kwargs['hidden_channels'],**special_keys)
            self.conv2 = GCNConv(kwargs['hidden_channels'], out_channels,**special_keys)
            # self.conv2 = GCNConv(kwargs['hidden_channels'], kwargs['hidden_channels'],**special_keys)
            # self.conv3 = GCNConv(kwargs['hidden_channels'], out_channels,**special_keys)
            self.reg_params = self.conv1.parameters
            self.non_reg_params = [self.conv2.parameters,self.lin1.parameters]
        
        if model == "GCN":
            
            self.conv1 = GCNConv(in_channels, kwargs['hidden_channels'],**special_keys)
            self.conv2 = GCNConv(kwargs['hidden_channels'], out_channels,**special_keys)
            self.reg_params = self.conv1.parameters
            self.non_reg_params = [self.conv2.parameters]
            
        elif model=="GAT":
            num_heads=kwargs['num_heads']
            head_dim = kwargs['hidden_channels']//num_heads
            hidden_channels=kwargs['hidden_channels']
            self.conv1 = GATConv(in_channels, head_dim, heads=num_heads)
            self.conv2 = GATConv(kwargs['hidden_channels'], out_channels, heads=1)
            self.reg_params = self.conv1.parameters
            self.non_reg_params = [self.conv2.parameters]
            
        elif model=="SAGE":
            hidden_channels=kwargs['hidden_channels']
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            self.reg_params = self.conv1.parameters
            self.non_reg_params = [self.conv2.parameters]

        elif model=="GIN":
            hidden_channels=kwargs['hidden_channels']
            self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

            self.conv1 = GINConv(self.lin1, train_eps=True)
            self.conv2 = GINConv(self.lin2, train_eps=True)
            self.reg_params = self.conv1.parameters
            self.non_reg_params = [self.conv2.parameters]

 

    def forward(self, input, edge_index,adj_matrix=None,phase=0,edge_weight=None,augment=False):
        special_keys={}

        if self.kwargs['ens']:
            special_keys['edge_weight'] = edge_weight
            special_keys['is_add_self_loops'] = True
        
        if self.kwargs['model_type']=='custom_GCN':
            x = self.lin1(input,edge_index)
            if self.kwargs['ens']:
                x, edge_index = x
            # x = functional.dropout(x, p= 0.5, training=self.training)
            
        else:
            x = input.clone()
        

        x = self.conv1(x, edge_index,**special_keys)
        
        if self.kwargs['ens']:
            x, edge_index = x
        
        x = functional.relu(x)
        x = functional.dropout(x, p= 0.5, training=self.training)
        x = self.conv2(x, edge_index,**special_keys)
        
        if self.kwargs['ens']:
            x, edge_index = x
        return x
