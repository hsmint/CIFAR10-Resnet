import torch.nn as nn
import torch.nn.functional as F

class AvgPoolPadding(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(AvgPoolPadding, self).__init__()
        self.identity = nn.AvgPool2d(stride, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, num_filter, channels_in=None, stride=1):
        super(ResBlock, self).__init__()

        if not channels_in or channels_in == num_filter:
            channels_in = num_filter
            self.projection = None
        else:
            self.projection = AvgPoolPadding(num_filter, channels_in, stride)

        self.conv1 = nn.Conv2d(channels_in, num_filter, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu2(out)
        return out

class BaseModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self):
        super(BaseModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers1 = self._make_layer(7, 16, 16, 1)
        self.layers2 = self._make_layer(7, 32, 16, 2)
        self.layers3 = self._make_layer(7, 64, 32, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, 10)

    def _make_layer(self, layer_count, channels, channels_in, stride):
        return nn.Sequential(
            ResBlock(channels, channels_in, stride),
            *[ResBlock(channels) for _ in range(layer_count-1)])

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
