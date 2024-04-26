import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SPADE, self).__init__()
        self.norm = nn.InstanceNorm2d(num_channels)
        self.gamma_conv = nn.Conv2d(num_classes, num_channels, kernel_size=3, padding=1)
        self.beta_conv = nn.Conv2d(num_classes, num_channels, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        gamma = self.gamma_conv(segmap)
        beta = self.beta_conv(segmap)

        normalized = self.norm(x)
        out = gamma * normalized + beta
        return out

# 示例用法
# 定义一个输入张量
input_tensor = torch.randn(1, 3, 64, 64)  # 假设输入为 64x64 的 RGB 图像
# 定义一个语义分割标签
segmap = torch.randint(0, 10, (1, 1, 64, 64))  # 假设有 10 个类别
# 初始化 SPADE 层
spade_layer = SPADE(num_channels=3, num_classes=10)
# 使用 SPADE 层进行归一化
output = spade_layer(input_tensor, segmap)
