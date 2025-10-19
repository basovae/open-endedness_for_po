# predictors.py
import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), output_activation=None, seed=None, **_):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.input_size = input_size          # <-- add this (optional but nice)
        self.output_size = output_size        # <-- add this (required)

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
