import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from association_graph import *

class general_model(object):
    def __init__(self, config, task_num, class_num):

        self.dataset = config["dataset"]
        self.file_out = config["file_out"]
        self.optim_param = config["lr"]
        self.task_num = task_num
        self.class_num = class_num
        self.beta = config["beta"]
        self.model = graph_model(config, task_num=self.task_num, class_num=self.class_num, backbone='res18').cuda()
        parameters =[{"params": self.model.parameters(), "lr": 1}]
        self.optimizer_list = optim.Adam(parameters, lr=1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
        self.criterion = nn.CrossEntropyLoss()
        self.iter_num = 1
        self.current_lr = 0.0

    def one_train_iteration(self, inputs_batch, labels_batch):
        if self.optimizer_list.param_groups[0]["lr"] >= 0.000002:
            self.current_lr = self.optim_param["init_lr"] * (self.optim_param["gamma"] ** (self.iter_num // self.optim_param["stepsize"]))
        for component in self.optimizer_list.param_groups:
            component["lr"] = self.current_lr * 1.0

        self.model.train()
        task_order = []
        for i in range(self.task_num):
            each_task_order = i * torch.ones(inputs_batch[i].shape[0], dtype=torch.int8, device="cuda")
            task_order.append(each_task_order)
        task_order = torch.cat(task_order)

        inputs_batch = torch.cat(inputs_batch, 0).cuda()
        labels_batch = torch.cat(labels_batch, 0).cuda()
        ae_loss, outputs, y_repeat = self.model(inputs_batch, labels_batch, task_order)
        mean_cls_loss = self.criterion(outputs, y_repeat)
        all_loss = mean_cls_loss + self.beta * ae_loss

        self.optimizer_list.zero_grad()
        all_loss.backward()
        self.optimizer_list.step()

        self.iter_num += 1

    def one_test_iteration(self, input, label, num):
        self.model.eval()
        task_order = torch.ones([input.shape[0]], dtype=torch.int8, device="cuda") * num
        _, output, y_repeat = self.model(input, label, task_order)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        _, output_predict = torch.max(output, 1)
        return output_predict, y_repeat