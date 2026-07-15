# src/recognition/models/sar_yolo.py
import torch
import torch.nn as nn

# ==========================================
# 1. 基础积木块：标准卷积层
# ==========================================
class ConvBlock(nn.Module):
    """标准的 YOLO 卷积块：Conv2d + BatchNorm + SiLU激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # 自动计算 padding，保证特征图大小不会因为卷积核大小而改变 (same padding)
        #就是为了保证经过卷积后特征图大小尺寸不变，same padding
        padding = kernel_size // 2 #取整
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)#bias=false表示不使用偏置
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() # SiLU 激活函数比 ReLU 更平滑，对噪声更有容忍度

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ==========================================
# 2. CBAM 注意力机制 (专门针对 SAR 散射点设计)
# ==========================================
class ChannelAttention(nn.Module):
    """
    通道注意力：决定'看什么'特征
    对于输入特征图的通道，神经网络自己决定哪些通道应该被关注，哪些应该被抑制
    输入：x，x.shape = [B,C,H,W]
    输出：每个通道一个权重，即[B,C,1,1]
    channels：输入特征图通道数
    reduction：降维比例，为了降低参数量而设置的；因为直接256->256参数很多，但是经过256->4->256参数量少
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #每个通道求平均：即每个通道的特征图都变成1*1大小了,全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1) #类似上面的全局平均池化，是全局最大池化
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),#inplace=True表示在原位置进行计算，不创建新的张量,节省显存
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        # 结合均值(背景信息)和最大值(强散射点信息)
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """
    空间注意力：决定'在哪里'看。这对 SAR 图像抑制背景杂波极其关键
    输入：x，x.shape = [B,C,H,W]
    输出：每个像素一个权重，即[B,1,H,W]
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)#输入通道数为2是因为最大和平均两个空间特征
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上取最大值和均值，突出高亮的散射点
        avg_out = torch.mean(x, dim=1, keepdim=True)# keepdim=false时，就会少了dim=1的那个维度；所以需要keepdim=true，保持维度
        max_out, _ = torch.max(x, dim=1, keepdim=True)# 只要最大值，不需要索引
        # 拼接后通过卷积生成空间注意力图
        pool_out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(pool_out))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)  # 先乘通道注意力
        x = x * self.sa(x)  # 再乘空间注意力
        return x

# ==========================================
# 3. 高级积木块：带注意力的 CSP 模块 (YOLO 的核心)
# ==========================================
class CSPBlock_SAR(nn.Module):
    """
    针对 SAR 改进的 CSP (Cross Stage Partial) 模块。
    原理：把输入劈成两半，一半去提特征，另一半直接保留。最后拼起来。
    这样既能减少计算量，又能防止 SAR 图像原本就微弱的特征在深层网络中丢失（梯度消失）。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        
        # 分支1：直接降维保留
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1)
        
        # 分支2：深度提取特征
        self.conv2 = ConvBlock(in_channels, mid_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            ConvBlock(mid_channels, mid_channels, kernel_size=3),
            ConvBlock(mid_channels, mid_channels, kernel_size=3)
        )
        # 融合后的注意力机制
        self.cbam = CBAM(out_channels)
        # 最后的输出融合
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.bottleneck(self.conv2(x))
        # 拼接特征
        out = torch.cat((y1, y2), dim=1)
        # 加入 CBAM 强行聚焦散射点
        out = self.cbam(out)
        return self.conv3(out)

# ==========================================
# 4. 主干网络组装：SAR-YOLO 分类网络
# ==========================================
class SARYoloCls(nn.Module):
    """
    完整的 SAR-YOLO 分类网络。
    结构：快速下采样 -> 多个针对SAR改进的CSP块提取特征 -> 全局池化 -> 分类输出
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Stem 层：快速降低分辨率，提取浅层边缘特征
        # SAR 图像是单通道灰度图，所以输入通道数为 1
        self.stem = nn.Sequential(
            ConvBlock(1, 32, kernel_size=3, stride=2),
            ConvBlock(32, 64, kernel_size=3, stride=2)
        )
        
        # 主干层：使用我们魔改的带 CBAM 的 CSP 模块
        self.layer1 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2),
            CSPBlock_SAR(128, 128)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2),
            CSPBlock_SAR(256, 256)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=2),
            CSPBlock_SAR(512, 512)
        )
        
        # 分类头 (Classification Head)
        # 无论输入多大的图，AdaptiveAvgPool2d(1) 都会把它变成 1x1，防止全连接层报错
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),#不管输入空间尺寸是多少，最终输出固定为 1×1。
            nn.Flatten(),
            nn.Dropout(0.3), # SAR 数据量小，加个 Dropout 防止过拟合
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.head(x)
        return out

# ==========================================
# 独立测试
# ==========================================
if __name__ == "__main__":
    # 模拟一张 SAR 图像输入：BatchSize=2, Channel=1 (灰度图), Height=128, Width=128
    dummy_input = torch.randn(2, 1, 128, 128)
    
    # 实例化网络 (我们有 10 个分类)
    model = SARYoloCls(num_classes=10)
    
    # 前向传播测试
    output = model(dummy_input)
    
    print(f"SAR-YOLO模型构建成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape} (应该输出 [2, 10])")