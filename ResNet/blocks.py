from copy import deepcopy
from layers import Conv2D, BatchNorm2D, ReLU
from layers import GlobalAvgPool, Linear
class BasicBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride)
            self.bn_down = BatchNorm2D(out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        identity = deepcopy(x)
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.downsample is not None:
            identity = self.downsample.forward(identity)
            identity = self.bn_down.forward(identity)

        out += identity
        out = self.relu.forward(out)
        return out