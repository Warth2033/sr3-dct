import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


# 检查对象是否存在
def exists(x):
    return x is not None


# 如果值存在，则返回该值，否则返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# 位置编码类 Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = (
            torch.arange(count, dtype=noise_level.dtype, device=noise_level.device)
            / count
        )
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


# 特征级别的仿射变换
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = (
                self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 上采样模块
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


# 下采样模块
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


# 基础块模块
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


# Resnet块模块
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        noise_level_emb_dim=None,
        dropout=0,
        use_affine_level=False,
        norm_groups=32,
    ):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level
        )

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


# 自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


# 带有注意力的Resnet块
class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        noise_level_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False
    ):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout
        )
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


# UNet网络结构
class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,  # 输入通道数，默认为6
        out_channel=3,  # 输出通道数，默认为3
        inner_channel=32,  # 内部通道数，即网络中间层的通道数，默认为32
        norm_groups=32,  # 归一化组数，用于GroupNorm层，默认为32
        channel_mults=(1, 2, 4, 8, 8),  # 通道倍数，用于定义各层的通道数，默认为(1, 2, 4, 8, 8)
        attn_res=(8),  # 使用注意力机制的分辨率，默认为8
        res_blocks=3,  # 残差块数量，默认为3
        dropout=0,  # dropout率，默认为0
        with_noise_level_emb=True,  # 是否使用噪声级别嵌入，默认为True
        image_size=128,  # 图像大小，默认为128
    ):
        super().__init__()  # 调用父类的初始化函数

        # 如果使用噪声级别嵌入
        if with_noise_level_emb:
            noise_level_channel = inner_channel  # 设置噪声级别通道数为内部通道数
            # 定义噪声级别的多层感知机，用于生成噪声级别嵌入
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),  # 位置编码层
                nn.Linear(inner_channel, inner_channel * 4),  # 线性层
                Swish(),  # Swish激活函数
                nn.Linear(inner_channel * 4, inner_channel),  # 线性层
            )
        else: # 如果不使用噪声级别嵌入
            noise_level_channel = None  # 则设置噪声级别通道数为None
            self.noise_level_mlp = None  # 设置噪声级别多层感知机为None


        # 定义下采样层
        num_mults = len(channel_mults)  # 获取通道倍数的长度
        pre_channel = inner_channel  # 设置前一个通道数为内部通道数
        feat_channels = [pre_channel]  # 初始化特征通道列表
        now_res = image_size  # 设置当前分辨率为图像大小
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]  # 初始化下采样层列表，添加第一个卷积层
        for ind in range(num_mults):  # 遍历通道倍数列表
            is_last = ind == num_mults - 1  # 判断是否是最后一个
            use_attn = now_res in attn_res  # 判断当前分辨率是否在注意力分辨率列表中
            channel_mult = inner_channel * channel_mults[ind]  # 计算当前层的通道数
            for _ in range(0, res_blocks):  # 添加残差块
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                    )
                )
                feat_channels.append(channel_mult)  # 添加当前层的通道数到特征通道列表
                pre_channel = channel_mult  # 更新前一个通道数
            if not is_last:  # 如果不是最后一个
                downs.append(Downsample(pre_channel))  # 添加下采样层
                feat_channels.append(pre_channel)  # 添加当前层的通道数到特征通道列表
                now_res = now_res // 2  # 更新当前分辨率
        self.downs = nn.ModuleList(downs)  # 将下采样层列表转换为模块列表

        # 定义中间层
        self.mid = nn.ModuleList(
            [
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=True,
                ),
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=False,
                ),
            ]
        )

        # 定义上采样层
        ups = []  # 初始化一个列表，用于存储上采样层
        for ind in reversed(range(num_mults)):  # 反向遍历通道倍数列表
            is_last = ind < 1  # 判断是否是最后一个
            use_attn = now_res in attn_res  # 判断当前分辨率是否在注意力分辨率列表中
            channel_mult = inner_channel * channel_mults[ind]  # 计算当前层的通道数
            for _ in range(0, res_blocks + 1):  # 添加残差块
                ups.append(
                    ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(),  # 输入通道数为前一个通道数加上特征通道列表的最后一个元素
                        channel_mult,  # 输出通道数
                        noise_level_emb_dim=noise_level_channel,  # 噪声级别嵌入维度
                        norm_groups=norm_groups,  # 归一化组数
                        dropout=dropout,  # dropout率
                        with_attn=use_attn,  # 是否使用注意力机制
                    )
                )
                pre_channel = channel_mult  # 更新前一个通道数
            if not is_last:  # 如果不是最后一个
                ups.append(Upsample(pre_channel))  # 添加上采样层
                now_res = now_res * 2  # 更新当前分辨率

        self.ups = nn.ModuleList(ups)  # 将上采样层列表转换为模块列表

        # 定义最终的卷积层
        self.final_conv = Block(
            pre_channel, default(out_channel, in_channel), groups=norm_groups
        )  # 最终的卷积层，使用Block模块

    # 前向传播函数
    def forward(self, x, time):
        # 如果存在噪声级别的多层感知机，则获取噪声级别
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        feats = []  # 初始化一个列表，用于存储各层的特征
        # 通过下采样层
        for layer in self.downs:  # 遍历下采样层列表
            if isinstance(layer, ResnetBlocWithAttn):  # 如果当前层是带有注意力的残差块
                x = layer(x, t)  # 通过当前层，并传入噪声级别嵌入
            else:
                x = layer(x)  # 否则直接通过当前层
            feats.append(x)  # 将当前层的输出添加到特征列表中

        # 通过中间层
        for layer in self.mid:  # 遍历中间层列表
            if isinstance(layer, ResnetBlocWithAttn):  # 如果当前层是带有注意力的残差块
                x = layer(x, t)  # 通过当前层，并传入噪声级别嵌入
            else:
                x = layer(x)  # 否则直接通过当前层

        # 通过上采样层
        for layer in self.ups:  # 遍历上采样层列表
            if isinstance(layer, ResnetBlocWithAttn):  # 如果当前层是带有注意力的残差块
                x = layer(torch.cat((x, feats.pop()), dim=1), t)  # 通过当前层，并传入噪声级别嵌入和之前存储的特征
            else:
                x = layer(x)  # 否则直接通过当前层

        # 通过最终的卷积层并返回结果
        return self.final_conv(x)
