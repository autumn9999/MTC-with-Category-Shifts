import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class N_layer_GNN(nn.Module):
    def __init__(self, GNN, num_layers, norm=None):
        super(N_layer_GNN, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(GNN) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, feat, adj):
        output = feat
        for GNN_layer in self.layers:
            output = GNN_layer(output, adj)
        if self.norm is not None:
            output = self.norm(output)
        return output

class MP_layer(nn.Module):
    def __init__(self, d_model):
        super(MP_layer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model * 2, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, feat, adj):
        x = feat
        node_size = adj.shape[1]
        I = torch.eye(node_size).cuda()
        adj = adj + I
        D = torch.diag(torch.sum(adj, 1))
        inverse_D = torch.linalg.inv(D)
        adj = torch.matmul(inverse_D, adj)

        pre_sup = self.linear1(x).squeeze(0)
        x1 = torch.matmul(adj, pre_sup).unsqueeze(0)
        x2 = self.norm1(torch.cat([x, x1], 2))
        output = self.norm2(self.linear2(x2))
        return output




