import torch
import numpy as np
import torch.nn as nn

def log_string(file_out, out_str, print_out=True):
    file_out.write(out_str+'\n')
    file_out.flush()
    if print_out:
        print(out_str)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        print('Unsupported value encountered.')


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.temperature = 32
        self.softmax = nn.Softmax(dim=2)
        self.project_queries = nn.Linear(d, d)
        self.project_keys = nn.Linear(d, d)
    def forward(self, queries, keys):
        queries = self.project_queries(queries)
        keys = self.project_keys(keys)
        attention = torch.bmm(queries, keys.transpose(1, 2))
        attention = self.softmax(attention / self.temperature)
        return attention



