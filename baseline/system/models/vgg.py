from collections import OrderedDict

import torch.nn as nn


class VGGa(nn.Module):
    def __init__(self, n_channels, n_classes, conv_params):
        super().__init__()

        conv_blocks = OrderedDict()
        for i in range(len(conv_params)):
            out_channels, pool_size, order = conv_params[i]
            in_channels = conv_params[i - 1][0] if i > 0 else n_channels
            conv_blocks[f'block{i}'] = conv_block_pooled(
                in_channels, out_channels, pool_size, order=order)

        self.bn = nn.BatchNorm2d(n_channels)
        self.conv_blocks = nn.Sequential(conv_blocks)
        self.classifier = nn.Linear(out_channels, n_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.embedding(x)  # (N, C)
        x = self.classifier(x)  # (N, K)
        return x

    def embedding(self, x):
        x = self.conv_blocks(x)  # (N, C, F', T')
        x = x.amax(dim=(2, 3))  # (N, C)
        return x


class VGG9a(VGGa):
    def __init__(self, n_channels, n_classes):
        conv_params = [
            (64, (2, 2), 2),
            (128, (2, 2), 2),
            (256, (2, 2), 2),
            (512, (2, 2), 2),
        ]
        super().__init__(n_channels, n_classes, conv_params)


class VGG11a(VGGa):
    def __init__(self, n_channels, n_classes):
        conv_params = [
            (64, (2, 2), 2),
            (128, (2, 2), 2),
            (256, (2, 2), 2),
            (512, (2, 2), 2),
            (512, (2, 2), 2),
        ]
        super().__init__(n_channels, n_classes, conv_params)


def conv_block_pooled(in_channels,
                      out_channels,
                      pool_size=(2, 2),
                      kernel_size=3,
                      order=2,
                      **kwargs,
                      ):
    layers = list(conv_block(in_channels, out_channels, kernel_size, **kwargs))
    for _ in range(order - 1):
        layers += conv_block(out_channels, out_channels, kernel_size, **kwargs)
    layers.append(nn.MaxPool2d(pool_size))

    return nn.Sequential(*layers)


def conv_block(in_channels, out_channels, kernel_size=3, **kwargs):
    padding = kernel_size // 2
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, bias=False, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return block
