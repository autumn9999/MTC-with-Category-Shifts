import os
import sys
import time
import math
import random
import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import models
import numpy as np

from MP_related import N_layer_GNN
from MP_related import MP_layer
from utils import Attention
resnet18 = models.resnet18(pretrained=True)

class graph_model(nn.Module):
    def __init__(self, config, task_num = 4, class_num=65, backbone='res18'):
        super(graph_model, self).__init__()

        self.task_num = task_num
        self.class_num = class_num
        self.backbone = backbone

        resnet = resnet18
        self.feature_dim = 512
        self.layer0 = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool
                    )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.layers=[self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        self.pool = resnet.avgpool

        special_dim = self.feature_dim
        self.task_representation = 0.0 * torch.zeros(self.task_num, special_dim).cuda()
        self.class_prototype = 0.0*torch.zeros(self.class_num, special_dim).cuda()
        encoder_layers = MP_layer(special_dim)
        self.N_layer_GNN = N_layer_GNN(encoder_layers, config["nlayers"]).cuda()
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.task_linear = nn.Linear(special_dim, 1)
        self.class_linear = nn.Linear(special_dim, 1)
        self.feature_edge = Attention(special_dim)
        self.base = nn.Linear(special_dim, self.class_num)
        self.task_specific_classifier_weight = nn.ParameterList(
            [nn.Parameter(torch.ones(self.class_num)) for i in range(self.task_num)])

    def statistics_extractor(self, x, labels_batch, task_order, extractor_type="task"):
        if extractor_type == "task":
            batch_type_representation = torch.zeros(self.task_num, self.feature_dim).cuda()
            for i in range(self.task_num):
                index_i = torch.where(task_order == i)[0]
                if len(index_i) != 0:
                    batch_type_representation[i] = x[index_i].mean(dim=[0, 2, 3]).unsqueeze(0)
                else:
                    continue

        elif extractor_type == "class":
            batch_type_representation = torch.zeros(self.class_num, self.feature_dim).cuda()
            for i in range(self.class_num):
                index_i = torch.where(labels_batch == i)[0]
                if len(index_i) != 0 :
                    batch_type_representation[i] = x[index_i].mean(dim=[0, 2, 3]).unsqueeze(0)
                else:
                    continue
        else:
            raise NotImplementedError
        return batch_type_representation

    def forward(self, x, labels_batch, task_order):
        # backbone
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        if self.training:
            # # updating the class prototype at each iteration during training
            batch_task_representation = self.statistics_extractor(x, labels_batch, task_order, extractor_type="task")
            self.task_representation = 0.9 * self.task_representation.detach() + 0.1 * batch_task_representation
            batch_class_prototype = self.statistics_extractor(x, labels_batch, task_order, extractor_type="class")
            self.class_prototype = 0.9 * self.class_prototype.detach() + 0.1 * batch_class_prototype
            c1 = self.class_prototype
            t1 = self.task_representation
        else:
            # test----------------
            c1 = self.class_prototype
            t1 = self.task_representation

        x1 = x.mean(dim=[2, 3])
        src = torch.cat([t1, c1, x1], 0).unsqueeze(0)

        # message passing
        adj, cross_graph = self.graph_adj(src)
        node_representation = self.N_layer_GNN(src, adj).squeeze(0)
        x2 = node_representation[self.task_num + self.class_num:]
        c2 = node_representation[self.task_num:self.task_num + self.class_num]
        t2 = node_representation[:self.task_num]

        # # updating the class and task by the graph
        if self.training:
            self.class_prototype = c2.detach()
            self.task_representation = t2.detach()

        # output
        y_repeat = labels_batch
        y_predict = self.base(x2)
        for i in range(self.task_num):
            index_i = torch.where(task_order == i)[0]
            y_predict[index_i] = self.task_specific_classifier_weight[i] * y_predict[index_i]

        negative_entropy = (cross_graph * torch.log(cross_graph)).sum(0).mean()
        ae_loss = negative_entropy

        return ae_loss, y_predict, y_repeat

    def graph_adj(self, x):
        t = x[:, 0:self.task_num].squeeze(0)
        c = x[:, self.task_num : self.task_num+self.class_num].squeeze(0)
        x_single = x[:, self.task_num+self.class_num:].squeeze(0)

        t_repeat = t.unsqueeze(1).repeat(1, self.class_num, 1)
        c_repeat = c.unsqueeze(0).repeat(self.task_num, 1, 1)
        alpha_p = 8.0
        cross_graph = self.softmax(-torch.sum(torch.square(t_repeat - c_repeat), -1) / (2.0 * alpha_p))

        task_graph = []
        for idx_i in range(self.task_num):
            tmp_dist = []
            for idx_j in range(self.task_num):
                if idx_i == idx_j:
                    dist = torch.zeros([1]).squeeze(0).cuda()
                else:
                    dist = self.sigmoid(self.task_linear(torch.abs(t[idx_i] - t[idx_j]))).squeeze(0)
                tmp_dist.append(dist)
            task_graph.append(torch.stack(tmp_dist))
        task_graph = torch.stack(task_graph)

        class_graph = []
        for idx_i in range(self.class_num):
            tmp_dist = []
            for idx_j in range(self.class_num):
                if idx_i == idx_j:
                    dist = torch.zeros([1]).squeeze(0).cuda()
                else:
                    dist = self.sigmoid(self.class_linear(torch.abs(c[idx_i] - c[idx_j]))).squeeze(0)
                tmp_dist.append(dist)
            class_graph.append(torch.stack(tmp_dist))
        class_graph = torch.stack(class_graph)

        adj = torch.concat([torch.concat([task_graph, cross_graph], axis=1),torch.concat([cross_graph.transpose(0,1), class_graph], axis=1)], axis=0)
        adj_x = self.feature_edge(x_single.unsqueeze(0), x)[0]
        cross_feat_graph = adj_x[:, :self.task_num + self.class_num]
        bz = x_single.shape[0]
        self_adj_x = torch.zeros([bz, bz]).cuda()
        adj_all = torch.concat([torch.cat([adj, cross_feat_graph.transpose(0, 1)], 1), torch.cat([cross_feat_graph, self_adj_x], 1)], 0)
        return adj_all, cross_graph
