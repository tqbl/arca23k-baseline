import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks):
        super().__init__()

        self.bn = nn.BatchNorm2d(n_channels)
        self.conv1 = conv3x3(n_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.blocks1 = _create_blocks(64, 64, n_blocks[0])
        self.blocks2 = _create_blocks(64, 128, n_blocks[1], pool_size=2)
        self.blocks3 = _create_blocks(128, 256, n_blocks[2], pool_size=2)
        self.blocks4 = _create_blocks(256, 512, n_blocks[3], pool_size=2)
        self.classifier = nn.Linear(512, n_classes)

        # Initialize weights of parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn(x)
        x = self.embedding(x)  # (N, C)
        x = self.classifier(x)  # (N, K)
        return x

    def embedding(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)

        x = x.amax(dim=(2, 3))  # (N, C)

        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_size=1,
                 downsample=None,
                 ):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size) if pool_size > 1 else None
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.pool is not None:
            x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.relu(x + identity)

        return x


class ResNet18a(ResNet):
    def __init__(self, n_channels, n_classes):
        super().__init__(n_channels, n_classes, [2, 2, 2, 2])


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, padding=1, bias=False)


def _create_blocks(in_channels, out_channels, n_blocks, pool_size=1):
    downsample = None
    if pool_size != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.MaxPool2d(pool_size),
            nn.BatchNorm2d(out_channels),
        )

    blocks = [BasicBlock(in_channels, out_channels, pool_size, downsample)]
    for _ in range(1, n_blocks):
        blocks.append(BasicBlock(out_channels, out_channels))

    return nn.Sequential(*blocks)
