from blocks import BasicBlock
from layers import Conv2D, BatchNorm2D, ReLU, GlobalAvgPool, Linear

class ResNet18Lite:
    def __init__(self, num_classes=10):
        self.in_channels = 16  # nhỏ hơn bản gốc
        self.conv1 = Conv2D(3, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2D(self.in_channels)
        self.relu = ReLU()

        # Các block có filter nhỏ hơn bản gốc
        self.layer1 = self._make_layer(BasicBlock, out_channels=16, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, out_channels=32, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, out_channels=64, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, out_channels=128, num_blocks=2, stride=2)

        self.pool = GlobalAvgPool()
        self.fc = Linear(in_features=128, out_features=num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return layers

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)

        for block in self.layer1: out = block.forward(out)
        for block in self.layer2: out = block.forward(out)
        for block in self.layer3: out = block.forward(out)
        for block in self.layer4: out = block.forward(out)

        out = self.pool.forward(out)
        out = out.reshape(out.shape[0], -1).T  # [C, B] cho Linear
        out = self.fc.forward(out)  # [num_classes, B]
        return out.T  # [B, num_classes]
