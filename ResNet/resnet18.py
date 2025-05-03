from layers import Conv2D, BatchNorm2D, ReLU, Linear, GlobalAvgPool
from blocks import BasicBlock
import numpy as np

class ResNet18:
    def __init__(self, num_classes=10):
        self.relu = ReLU()
        self.conv1 = Conv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2D(64)
        self.layer1 = [BasicBlock(64, 64) for _ in range(2)]
        self.layer2 = [BasicBlock(64, 128, stride=2)] + [BasicBlock(128, 128) for _ in range(1)]
        self.layer3 = [BasicBlock(128, 256, stride=2)] + [BasicBlock(256, 256) for _ in range(1)]
        self.layer4 = [BasicBlock(256, 512, stride=2)] + [BasicBlock(512, 512) for _ in range(1)]
        self.pool = GlobalAvgPool()
        self.fc = Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)
        for block in self.layer1:
            out = block.forward(out)
        for block in self.layer2:
            out = block.forward(out)
        for block in self.layer3:
            out = block.forward(out)
        for block in self.layer4:
            out = block.forward(out)
        out = self.pool.forward(out)
        out = out.reshape(out.shape[0], -1).T  # (batch_size, features) --> (features, batch_size)
        out = self.fc.forward(out)
        return out.T  # trả về (batch_size, num_classes)