# src/recognition/models/sar_vit.py
import torch
import torch.nn as nn

# ==========================================
# 1. 针对 SAR 改进的 CPE (Convolutional Patch Embedding)
# ==========================================
class ConvPatchEmbedding(nn.Module):
    """
    不用传统的 Linear 暴力切块，而是用连续的卷积进行“软提取”。
    能够有效保留 SAR 图像中离散亮斑的局部几何拓扑关系。
    输入：[B,C,H,W]->[B,1,128,128]
    输出：[B,H/8*H/8,embed_dim]->[B,256,embed_dim],B代表batchsize，256为token数量，embed_dim是token高维嵌入维度
    """
    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        # 第一步：7x7 大卷积核，步长 4。抑制散斑噪声，提取粗特征
        # 输入 128x128 -> 输出 32x32，下采样4倍
        self.proj1 = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.act1 = nn.GELU()
        
        # 第二步：3x3 卷积核，步长 2。进一步下采样
        # 输入 32x32 -> 输出 16x16，下采样2倍
        self.proj2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.proj1(x)))
        x = self.bn2(self.proj2(x))
        # 此时 x 的形状是 (Batch, embed_dim, 16, 16)
        # Transformer 需要序列，所以我们把它展平并转置 -> (Batch, 256, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

# ==========================================
# 2. 标准的 Transformer 编码器块
# ==========================================
class TransformerBlock(nn.Module):
    """ Transformer 核心模块：自注意力机制 + 前馈神经网络
    采用的Pre-Norm Transformer，训练更稳定，深层更容易
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)#对最后一维进行归一化，也就是对每个token的高维嵌入表示自身进行归一化
        # 多头自注意力 (Multi-Head Self Attention)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        #关于上面多头注意力的参数设置，batch_first=True 
        #Pytorch的默认维度顺序是(sequence,batch,embedding)
        # batch_first=True则表示输入和输出的维度顺序是 (batch, sequence, embedding)

        self.norm2 = nn.LayerNorm(embed_dim)
        #为了给MLP的隐藏层维度
        hidden_dim = int(embed_dim * mlp_ratio)
        # 前馈网络 (MLP)
        '''Linear
            |
            GELU
            |
            Dropout
            |
            Linear
            |
            Dropout
        '''
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 带有残差连接的 Attention，自注意力，Q,K,V都是x
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # 带有残差连接的 MLP
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 3. 组装 SAR-ViT
# ==========================================
class SARViT(nn.Module):
    """
    为 SAR 图像定做的小型 Vision Transformer。
    包含：卷积软切块 -> 绝对位置编码 -> 类别 Token -> 多层 Transformer -> 分类头
    """
    def __init__(self, num_classes=10, embed_dim=256, depth=2, num_heads=8):
        super().__init__()
        
        # 1. 切块与嵌入 (Patch Embedding)
        self.patch_embed = ConvPatchEmbedding(in_channels=1, embed_dim=embed_dim)
        
        # 对于 128x128 的图，经过 stride=8 的卷积后，特征图是 16x16 = 256 个 token
        num_patches = 16 * 16
        
        # 2. 分类 Token 与位置编码 (Positional Encoding)
        # cls_token 是用来最终代表整张图信息的向量
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 是因为多了一个 cls_token
        #位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=0.1)
        
        # 3. Transformer 堆叠层 (Depth 代表有几个 Block)
        # SAR 数据少，深度不能太深，2层足以，否则严重过拟合
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. 最终的分类头
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        
        # 提取 Patch 特征
        x = self.patch_embed(x)  # (B, 256, Embed_Dim)
        
        # 拼接 cls_token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, Embed_Dim)
        x = torch.cat((cls_tokens, x), dim=1)         # (B, 257, Embed_Dim)
        
        # 加上位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 穿过所有 Transformer 层
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        # 提取第 0 个位置的 token (也就是 cls_token) 作为全局图像特征
        #取 CLS token → [B, Embed_Dim]
        cls_feat = x[:, 0] 
        
        # 分类输出
        out = self.head(cls_feat)
        return out

# ==========================================
# 独立测试
# ==========================================
if __name__ == "__main__":
    dummy_input = torch.randn(2, 1, 128, 128)
    model = SARViT(num_classes=10)
    output = model(dummy_input)
    
    print(f"SAR-ViT模型构建成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape} (应该输出 [2, 10])")