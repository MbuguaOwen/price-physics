try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - allow import without heavy deps
    torch = None
    nn = None
    F = None
if torch is not None and nn is not None and F is not None:
    class BasicBlock(nn.Module):
        def __init__(self, c_in, c_out, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(c_out)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(c_out)
            self.shortcut = nn.Sequential()
            if stride != 1 or c_in != c_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(c_out)
                )
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNetLite(nn.Module):
        def __init__(self, num_classes=3, in_ch=4):
            super().__init__()
            self.layer1 = BasicBlock(in_ch, 16, stride=2)
            self.layer2 = BasicBlock(16, 32, stride=2)
            self.layer3 = BasicBlock(32, 64, stride=2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
            x = self.pool(x).squeeze(-1).squeeze(-1)
            return self.fc(x)

    def build_model(num_classes=3, in_ch=4):
        return ResNetLite(num_classes=num_classes, in_ch=in_ch)
else:
    class ResNetLite:  # minimal stub to allow import without torch
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for ResNetLite; please install torch to use this model.")

    def build_model(num_classes=3, in_ch=4):
        raise ImportError("torch is required for build_model; please install torch.")
