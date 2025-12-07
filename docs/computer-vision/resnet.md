# Deep Residual Learning for Image Recognition (ResNet)

<div class="paper-meta" markdown>

**作者**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**机构**: Microsoft Research
**会议**: CVPR 2016
**论文链接**: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">ResNet</span>
<span class="paper-tag">残差连接</span>
<span class="paper-tag">图像分类</span>
<span class="paper-tag">深度网络</span>
</div>

## 📝 摘要

本文提出了深度残差学习框架（ResNet），通过引入残差连接（skip connection）解决了深度神经网络的退化问题。ResNet 在 ImageNet 分类、COCO 检测和分割等任务上取得了显著提升，并赢得了 ILSVRC 2015 图像分类冠军。

**主要贡献**：
- 提出残差学习框架，使训练超过 100 层的网络成为可能
- 在 ImageNet 上以 152 层网络达到 3.57% top-5 错误率
- 证明了极深网络的有效性

## 💡 核心思想

### 退化问题（Degradation Problem）

实验发现，简单地堆叠更多层会导致**退化**：更深的网络训练误差反而更高（不是过拟合）。

### 残差学习（Residual Learning）

与其学习直接映射 $H(x)$，不如学习残差映射：

$$
F(x) = H(x) - x
$$

原始映射变为：

$$
H(x) = F(x) + x
$$

**直觉**：学习"如何调整"比学习"完整输出"更容易。

### 残差块（Residual Block）

```
x → [Conv-BN-ReLU-Conv-BN] → F(x)
 ↓                              ↓
 └──────────────────────────→ (+) → ReLU → output
           identity
```

## 🏗️ 网络架构

### ResNet-50/101/152

| 层名 | 输出尺寸 | ResNet-50 | ResNet-101 | ResNet-152 |
|------|----------|-----------|------------|------------|
| conv1 | 112×112 | 7×7, 64, stride 2 | 7×7, 64, stride 2 | 7×7, 64, stride 2 |
| conv2_x | 56×56 | 3×3 max pool, stride 2<br>[1×1,64<br>3×3,64<br>1×1,256] ×3 | ×3 | ×3 |
| conv3_x | 28×28 | [1×1,128<br>3×3,128<br>1×1,512] ×4 | ×4 | ×8 |
| conv4_x | 14×14 | [1×1,256<br>3×3,256<br>1×1,1024] ×6 | ×23 | ×36 |
| conv5_x | 7×7 | [1×1,512<br>3×3,512<br>1×1,2048] ×3 | ×3 | ×3 |
| | 1×1 | average pool, 1000-d fc, softmax | | |

### 瓶颈结构（Bottleneck）

为了降低计算量，ResNet-50/101/152 使用瓶颈设计：

```
1×1 conv (降维) → 3×3 conv → 1×1 conv (升维)
```

## 📊 实验结果

### ImageNet 分类

| 模型 | Top-1 Error | Top-5 Error | 参数量 |
|------|-------------|-------------|--------|
| VGG-16 | - | 7.3% | 138M |
| GoogleNet | - | 6.7% | - |
| **ResNet-34** | 24.5% | - | 21.8M |
| **ResNet-50** | 22.9% | - | 25.6M |
| **ResNet-101** | 21.8% | - | 44.5M |
| **ResNet-152** | **21.4%** | **3.57%** | 60.2M |

### COCO 目标检测

基于 Faster R-CNN + ResNet-101：

- **mAP**: 提升 6% 相比 VGG-16

### 消融实验

**恒等映射 vs 投影**：
- 恒等 shortcut 效果最好
- 投影 shortcut 略有提升但增加参数

**网络深度**：
- 18层 → 34层：错误率显著下降
- 50层 → 101层 → 152层：持续改进

## 🤔 个人笔记

### 为什么残差连接有效？

1. **梯度流动**：shortcut 提供了梯度的直接通路，缓解梯度消失
2. **集成效应**：网络可以看作多个浅层网络的集成
3. **优化景观**：残差学习使损失函数更平滑，更容易优化

### 设计要点

- **批归一化（BN）**：每个卷积后都使用 BN
- **无 Dropout**：ResNet 不使用 dropout
- **数据增强**：标准的裁剪、翻转、颜色抖动

### 后续发展

- **ResNeXt**：引入分组卷积，提升表示能力
- **DenseNet**：密集连接，进一步加强特征复用
- **EfficientNet**：通过 NAS 自动搜索高效架构
- **ResNet 变体**：Pre-activation ResNet、Wide ResNet 等

### 影响

ResNet 的残差连接思想影响深远：

- **Transformer**：也使用残差连接
- **目标检测**：FPN、Mask R-CNN 等都基于 ResNet
- **工业应用**：ResNet 成为计算机视觉的标准骨干网络

## 🔗 相关论文

- **VGG**: Very Deep Convolutional Networks - 更早的深度网络尝试
- **Inception/GoogleNet**: Going Deeper with Convolutions - 多尺度卷积
- **ResNeXt**: Aggregated Residual Transformations - ResNet 改进
- **DenseNet**: Densely Connected Convolutional Networks - 密集连接

## 📌 关键代码片段

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet-18/34 的基本残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet-50/101/152 的瓶颈残差块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
```

---

*阅读日期：2024年*
*笔记状态：已完成*
