import torch.nn as nn


class BaseNetwork(nn.Module):
    def print_network(self):
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        print(
            f'Network [{type(self).__name__}] has '
            f'{parameter_count / 1000000:.1f} million parameters.'
        )

    def init_weights(self, init_type='normal', gain=0.02):
        def initialize(module):
            name = module.__class__.__name__
            if 'InstanceNorm2d' in name:
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif hasattr(module, 'weight') and (
                    'Conv' in name or 'Linear' in name):
                if init_type == 'normal':
                    nn.init.normal_(module.weight, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(module.weight, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(
                        module.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(module.weight, gain=gain)
                elif init_type == 'none':
                    module.reset_parameters()
                else:
                    raise ValueError(f'Unknown initialization: {init_type}')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.apply(initialize)
        for child in self.children():
            if hasattr(child, 'init_weights'):
                child.init_weights(init_type, gain)
