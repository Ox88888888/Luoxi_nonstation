import torch
import numpy as np
import matplotlib.pyplot as plt
from models import generators  # 导入生成模型类
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def color_replace(img_tensor):
    # 将张量的值从 [0, 1] 映射到 [0, 255]，并转换为 numpy 数组
    img_np = (img_tensor * 255).clamp(0, 255).byte().numpy()

    # 定义颜色替换映射
    deep_purple_color = np.array([75, 0, 130])  # 深紫色 RGB 值
    yellow_color = np.array([255, 255, 0])  # 黄色 RGB 值

    # 根据图像像素值进行颜色替换
    img_rgb = img_np.copy()

    # 定义更大范围的黑色和白色的像素值范围
    black_range = (0, 150)  # 更大的黑色像素值范围
    white_range = (100, 255)  # 更大的白色像素值范围

    # 替换黑色为深紫色
    black_mask = np.all((img_np >= black_range[0]) & (img_np <= black_range[1]), axis=-1)
    img_rgb[black_mask] = deep_purple_color

    # 替换白色为黄色
    white_mask = np.all((img_np >= white_range[0]) & (img_np <= white_range[1]), axis=-1)
    img_rgb[white_mask] = yellow_color

    return img_rgb

# 设置设备，CPU 或 GPU
device = torch.device("cpu")


# 创建与预训练模型相同结构的新模型
netG = generators.Res_Generator(z_dim=128, n_classes=4, base_ch=52, att=True, img_ch=3, leak=0, cond_method='conv1x1', SN=False).to(device)

# 加载预训练模型参数，但只加载匹配的部分参数
checkpoint = torch.load('D:/LuoXi_file/NonstationaryGANs/trained_models/BR_model.pth', map_location=device)
state_dict = checkpoint['netG_state_dict']

# 修正参数名以匹配新模型
new_state_dict = {}
for key, value in state_dict.items():
    # 如果不是最终层的参数，直接加载
    if 'final' not in key:
        new_state_dict[key] = value
    else:
        # 对于最终层的参数，根据新模型的形状调整
        if 'weight' in key:
            # 扩展预训练模型的权重形状为多通道
            weight_expanded = value.expand(netG.final.weight.shape)
            new_state_dict[key] = weight_expanded
        elif 'bias' in key:
            # 复制多个偏置值以匹配新模型的偏置形状
            bias_expanded = value.expand(netG.final.bias.shape)
            new_state_dict[key] = bias_expanded

# 加载修正后的参数
netG.load_state_dict(new_state_dict)

# 设置模型为评估模式
netG.eval()

# 生成条件张量 y
y = torch.tensor([[0.7690, 0.7975, 0.1910, 0.5455],
                  [0, 0.2695, 0, 0.9895],
                  [0, 0.1667, 0, 0.4514],
                  [0, 0.9624, 0.6627, 0.6666]]).unsqueeze(0).unsqueeze(0).to(device)

plt.imshow(y[0,0])
plt.colorbar()
plt.axis('off')
plt.clim(0, 1)

# 生成随机噪声张量 z
N = 2
z = torch.randn(N**2, 128).to(device)

# 使用生成模型 netG 生成图像
with torch.no_grad():
    imgs = netG(z, y).cpu()  # 注意这里不需要调整范围和转换为 numpy 数组

# 可视化生成的彩色图像（进行颜色替换）
fig, axes = plt.subplots(N, N, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    # 将图像进行颜色替换并显示
    img_rgb = color_replace(imgs[i].clamp(0, 1).permute(1, 2, 0))  # 将 clamp 移到前面确保范围在 [0, 1] 内
    ax.imshow(img_rgb)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


            

