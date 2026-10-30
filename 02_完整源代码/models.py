"""基础 CNN 和经典 SE 通道注意力，所有张量形状见 forward。"""
import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, max(1, channels // reduction), bias=False),
                                nn.ReLU(), nn.Linear(max(1, channels // reduction), channels, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        weights = self.fc(self.pool(x).flatten(1))[:, :, None, None]
        return x * weights


class ClothingCNN(nn.Module):
    def __init__(self, use_se=False, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.se1 = SEBlock(32, reduction) if use_se else nn.Identity()
        self.se2 = SEBlock(64, reduction) if use_se else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.se1(self.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.se2(self.relu(self.bn2(self.conv2(x)))))
        return self.fc2(self.dropout(self.relu(self.fc1(x.flatten(1)))))


def make_model(use_se, seed, reduction=8):
    # 先建立相同 baseline，再复制共享层，防止 SE 初始化改变分类层初始值。
    torch.manual_seed(seed)
    baseline = ClothingCNN(False, reduction)
    if not use_se:
        return baseline
    torch.manual_seed(seed + 100000)
    model = ClothingCNN(True, reduction)
    model.load_state_dict(baseline.state_dict(), strict=False)
    return model


def count_macs(model):
    """单图 Conv/Linear 乘加次数；不含 BN、激活、池化和逐元素乘法。"""
    total = [0]
    def hook(layer, inputs, output):
        if isinstance(layer, nn.Conv2d):
            total[0] += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, nn.Linear):
            total[0] += output.numel() * layer.in_features
    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    state = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 1, 28, 28, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    model.train(state)
    return total[0]
