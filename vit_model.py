"""
Vision Transformer (ViT) 模型实现
原始代码来自 rwightman 的 pytorch-image-models：
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
参考论文：
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (https://arxiv.org/abs/2010.11929)
"""

from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

# -----------------------------------
# Drop Path (随机深度) 相关实现
# -----------------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    对输入的张量 x 应用随机深度（DropPath），即随机丢弃部分残差分支，
    有助于正则化模型和缓解过拟合。
    
    参数：
        x: 输入张量
        drop_prob: 丢弃概率
        training: 是否处于训练模式
        
    返回：
        经过随机深度处理后的张量
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # 创建一个 shape 为 (batch_size, 1, 1, ...) 的随机张量
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化：保留 1 或丢弃 0
    output = x.div(keep_prob) * random_tensor  # 保证期望不变
    return output

class DropPath(nn.Module):
    """
    封装 drop_path 函数，作为一个 nn.Module 使用。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------------
# Patch Embedding 模块
# -----------------------------------
class PatchEmbed(nn.Module):
    """
    将输入图像切分成若干个固定大小的 Patch，并对每个 Patch 进行线性投影，
    将 2D 图像转换为序列形式，便于 Transformer 处理。
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)         # 将图像尺寸转为 tuple
        patch_size = (patch_size, patch_size)     # 将 Patch 尺寸转为 tuple
        self.img_size = img_size
        self.patch_size = patch_size
        # grid_size: 图像在高度和宽度方向上可分为多少个 Patch
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 总的 Patch 数量
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 使用 Conv2d 以 patch_size 大小的卷积核、步幅为 patch_size 来实现切分和投影
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 可选归一化层
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        x: 输入张量，形状 [B, C, H, W]
        返回：
            Patch Embedding，形状 [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图片尺寸 ({H}*{W}) 与模型要求 ({self.img_size[0]}*{self.img_size[1]}) 不匹配。"
        # 通过卷积得到形状 [B, embed_dim, num_patches_H, num_patches_W]，
        # flatten(2) 将最后两维展平 -> [B, embed_dim, num_patches]，
        # transpose 将通道维移到最后 -> [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# -----------------------------------
# Attention 模块
# -----------------------------------
class Attention(nn.Module):
    def __init__(self,
                 dim,                # 输入 token 的总维度
                 num_heads=8,        # 注意力头数
                 qkv_bias=False,     # 是否使用偏置
                 qk_scale=None,      # 缩放因子
                 attn_drop_ratio=0., # 注意力 dropout 率
                 proj_drop_ratio=0.  # 输出投影 dropout 率
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子
        # 通过一个线性层同时生成 Query, Key, Value，输出维度为 3 * dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        """
        x: 输入 tensor，形状 [B, N, C]，其中 N 是 token 数量 (例如，patch 数量加 1 个 CLS token)
        返回：
            注意力模块的输出，形状 [B, N, C]
        """
        B, N, C = x.shape
        # 生成 q, k, v：先通过 qkv 线性层，输出形状为 [B, N, 3 * C]
        # reshape 为 [B, N, 3, num_heads, C // num_heads]，再 permute 得到 [3, B, num_heads, N, C // num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 分离得到 query, key, value，每个形状 [B, num_heads, N, C // num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力权重：q @ k^T，得到形状 [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 用注意力权重乘上 v：输出形状 [B, num_heads, N, C // num_heads]
        x = (attn @ v)
        # 将多头拼接回去：先 transpose，形状变为 [B, N, num_heads, C // num_heads]，再 reshape 为 [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------------
# MLP 模块
# -----------------------------------
class Mlp(nn.Module):
    """
    多层感知机模块，通常用于 Transformer 中，
    包括两层全连接层，中间有激活函数（GELU）和 dropout。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 第一层全连接：扩展特征维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # 第二层全连接：还原维度
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -----------------------------------
# Transformer Block (Encoder Block)
# -----------------------------------
class Block(nn.Module):
    """
    Transformer Block，由两个子层组成：
      1. 多头自注意力层（带残差连接和 DropPath）
      2. MLP 模块（带残差连接和 DropPath）
    每个子层前都使用 LayerNorm。
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # 随机深度 (DropPath) 正则化
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # 先对输入进行 LayerNorm，再进入自注意力层，之后加上原输入（残差连接）并应用 DropPath
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # 对上一步结果进行 LayerNorm，再通过 MLP 模块，残差连接和 DropPath
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------------
# Vision Transformer 主模型
# -----------------------------------
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Vision Transformer 初始化函数
        参数：
            img_size: 输入图像尺寸
            patch_size: Patch 尺寸，用于切分图像
            in_c: 图像输入通道数
            num_classes: 分类头的类别数
            embed_dim: 每个 Patch 经过线性投影后的维度
            depth: Transformer Block 的堆叠层数
            num_heads: 注意力头数
            mlp_ratio: MLP 模块中隐藏层维度与 embed_dim 的比例
            qkv_bias: 是否在 qkv 线性层中使用偏置
            qk_scale: 缩放因子
            representation_size: 如果提供，则会在最后加入一个 representation 层
            distilled: 是否使用蒸馏 token（如 DeiT 模型）
            drop_ratio: dropout 率
            attn_drop_ratio: 注意力 dropout 率
            drop_path_ratio: 随机深度（DropPath）率
            embed_layer: 用于 Patch Embedding 的模块（默认 PatchEmbed）
            norm_layer: 归一化层（默认使用 LayerNorm，eps=1e-6）
            act_layer: 激活函数（默认 GELU）
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # 保证后续一致性
        # 根据是否使用 distillation token，决定总 token 数（CLS token 或 CLS + dist_token）
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Patch embedding 模块，将图像转换为 Patch token 序列
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token，为模型学习一个全局表示
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # dist_token（可选），用于蒸馏策略
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # 位置编码，为所有 token 分配可学习的位置向量
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 为每个 Block 设置一个不同的 DropPath 概率，按线性衰减分布
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # 堆叠多个 Transformer Block
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        # 最后一个归一化层
        self.norm = norm_layer(embed_dim)

        # 可选的 representation 层（pre-logits）
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 分类头：将提取的特征映射到类别数目
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化：初始化位置编码、CLS token、dist_token 及其它模块参数
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        """
        提取输入图像的特征
        参数：
            x: 输入图像张量，形状 [B, C, H, W]
        返回：
            如果不使用 distillation token，返回经过预处理的 CLS token 的特征
            如果使用，则返回 CLS token 和 dist_token 的特征
        """
        # 将图像转换为 patch token 序列，形状 [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # 例如 [B, 196, 768]
        # 扩展 CLS token，使其与批次大小匹配 [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 拼接 CLS token 和 patch tokens；如果有 dist_token，则先拼接 CLS，再拼接 dist_token
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, embed_dim]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # 将位置编码加到 token 序列上，并进行 dropout
        x = self.pos_drop(x + self.pos_embed)
        # 经过堆叠的 Transformer Blocks
        x = self.blocks(x)
        # 最后归一化
        x = self.norm(x)
        # 如果不使用 distillation token，则返回第一个 token（CLS）的特征，经过 pre_logits 处理
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        """
        前向传播，计算分类 logits
        参数：
            x: 输入图像张量 [B, C, H, W]
        返回：
            分类的 logits，若使用 distillation token，则在推理时返回两头平均值
        """
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    初始化 ViT 模块的权重
    参数：
        m: 模块
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# -----------------------------------
# 定义不同版本的 ViT 模型创建函数
# -----------------------------------
def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base (ViT-B/16) 模型，用于 ImageNet-1k 分类。
    预训练权重见论文说明及链接。
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base (ViT-B/16) 模型，用于 ImageNet-21k 预训练。
    预训练权重链接：
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base (ViT-B/32) 模型，用于 ImageNet-1k 分类。
    预训练权重链接和密码见代码注释。
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base (ViT-B/32) 模型，用于 ImageNet-21k 预训练。
    预训练权重链接：
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large (ViT-L/16) 模型，用于 ImageNet-1k 分类。
    预训练权重链接和密码见代码注释。
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large (ViT-L/16) 模型，用于 ImageNet-21k 预训练。
    预训练权重链接：
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large (ViT-L/32) 模型，用于 ImageNet-21k 预训练。
    预训练权重链接：
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge (ViT-H/14) 模型，用于 ImageNet-21k 预训练。
    注意：由于文件过大，转换后的权重暂不可用。
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
