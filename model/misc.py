import torch.nn as nn


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)
