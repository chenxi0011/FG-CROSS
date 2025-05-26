from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image


class Bottleneck(nn.Module):
    '''
    1. 作用
    瓶颈层：Bottleneck 是一种特殊的残差块，用于减少计算量和参数数量，同时保持或提高模型的性能。
    特征提取：通过多层卷积操作，逐步压缩和扩展特征图，从而更高效地提取特征。
    2. 结构
    1x1 卷积：第一个 1x1 卷积层用于降维，减少通道数，降低计算复杂度。
    3x3 卷积：中间的 3x3 卷积层用于捕捉空间特征。
    1x1 卷积：最后一个 1x1 卷积层用于升维，恢复通道数，与输入特征图的通道数匹配。
    3. 残差连接
    跳过连接：Bottleneck 类通常包含一个跳过连接（skip connection），将输入直接加到输出上，有助于缓解梯度消失问题，提高网络的训练效果。

    # 测试
    bottleneck = Bottleneck(inplanes=64, planes=64, stride=2)
    x = torch.randn(1, 64, 56, 56)  # 输入大小为 [batch_size, channels, height, width]
    output = bottleneck(x)
    print(output.shape)  # 输出大小为 [1, 256, 28, 28]
    如果 downsample 为 None，则 identity 保持不变，形状为 (1, 64, 56, 56)。为了与 out 相加，downsample 需要将 identity 的通道数从 64 变为 256。
    如果 downsample 不为 None，则 identity 会被调整为 (1, 256, 56, 56)
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    '''
    注意力池化：AttentionPool2d 类通过注意力机制对特征图进行池化操作，使得模型能够关注特征图中更重要的区域，从而提高模型的性能。
    动态加权：与传统的池化方法（如最大池化、平均池化）不同，注意力池化通过对每个位置的特征进行加权，使得重要的特征被赋予更高的权重.

    初始化：
        spacial_dim：输入特征图的通道数。
        embed_dim：输出特征图的通道数。
        pool_size：池化窗口的大小。
        heads：注意力头的数量（默认为 1）。
        query、key、value：线性变换层，用于生成查询、键和值。
        positional_embedding：位置嵌入参数，用于添加位置信息。
    前向传播：
        展平：将输入特征图展平为 (B, N, C) 形状，其中 N = H * W。
        线性变换：对展平后的特征图进行线性变换，生成查询、键和值。
        位置嵌入：将位置嵌入添加到查询和键中，以引入位置信息。
        注意力权重：计算注意力权重矩阵，并使用 softmax 函数进行归一化。
        加权求和：将注意力权重应用于值，得到加权后的特征图。
        重塑输出：将加权后的特征图重塑为 (B, out_channels, H, W) 形状。

    # 定义 AttentionPool2d 实例
    spacial_dim = 7  # 特征图的空间维度
    embed_dim = 512  # 嵌入维度
    num_heads = 8  # 注意力头数
    output_dim = 512  # 输出维度
    attention_pool = AttentionPool2d(spacial_dim, embed_dim, num_heads, output_dim)
    # 准备输入数据
    x = torch.randn(1, 512, 7, 7)
    # 前向传播
    output = attention_pool(x)
    print(output.shape)  # 输出形状为 [1, 512]
    '''
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    三阶段的 stem 层：使用三个卷积层而不是一个卷积层，并且在 stem 层后使用平均池化而不是最大池化。
    抗混叠下采样：在步长大于 1 的卷积层之前添加了一个平均池化层，以减少混叠效应。
    最终的注意力池化层：用 QKV 注意力池化层替换了传统的全局平均池化层。

    在 CLIP 模型中，ModifiedResNet 作为图像编码器的一部分，负责将输入图像转换为固定长度的特征向量。
    具体来说，ModifiedResNet 被定义在 CLIP 类的初始化方法中，当配置中指定了使用 ResNet 作为视觉模型时，
    会实例化 ModifiedResNet。

    定义 ModifiedResNet 实例：
    layers：ResNet 的层数配置，例如 [3, 4, 6, 3] 表示 ResNet50。
    output_dim：最终输出的特征维度。
    heads：多头注意力机制中的头数。
    input_resolution：输入图像的分辨率。
    width：初始宽度，即第一个卷积层的输出通道数。
    准备输入数据：
    x：随机生成的输入图像，形状为 [1, 3, 224, 224]。
    前向传播：
    modified_resnet(x)：调用 ModifiedResNet 的前向传播方法，处理输入图像。
    output：最终的输出特征，形状为 [1, 512]，表示一批次大小为 1，特征维度为 512 的输出。

    # 定义 ModifiedResNet 实例
    layers = [3, 4, 6, 3]  # ResNet50 的层数配置
    output_dim = 512  # 输出特征维度
    heads = 8  # 注意力头数
    input_resolution = 224  # 输入图像分辨率
    width = 64  # 初始宽度
    modified_resnet = ModifiedResNet(layers, output_dim, heads, input_resolution, width)

    # 准备输入数据
    x = torch.randn(1, 3, 224, 224)
    # 前向传播
    output = modified_resnet(x)
    print(output.shape)  # 输出形状为 [1, 512]
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16.
    通过自定义 LayerNorm 的前向传播方法处理输入数据，最终输出一个形状为 [1, 512] 的归一化特征向量
    数据类型仍为半精度浮点数（fp16）
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    '''
    QuickGELU 通过使用 sigmoid 函数来近似 GELU，从而在保持性能的同时提高计算效率
    定义 QuickGELU 实例：
    quick_gelu = QuickGELU()：创建一个 QuickGELU 实例。
    准备输入数据：
    x：随机生成的输入数据，形状为 [1, 512]。
    前向传播：
    quick_gelu(x)：调用 QuickGELU 的前向传播方法，处理输入数据。
    output：最终的输出特征，形状为 [1, 512]，表示一批次大小为 1，特征维度为 512 的激活特征向量
    '''
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    '''
    这个类是 Transformer 的基本组成模块，它实现了基于多头注意力的残差块。
    多头注意力机制 (MultiheadAttention)：捕获序列中不同元素之间的全局依赖关系。
    残差连接 (Residual)：通过加法操作将输入直接与处理后的结果相加，有助于梯度传播。
    前馈网络 (MLP)：增强模型的非线性特征建模能力。
    归一化层 (LayerNorm)：稳定训练，避免梯度爆炸或消失。
    '''
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # n_head 头，d_model 表示维度。
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]  # 三个x表示Q K V计算值，x最后维度=n_head*d_model

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    '''
    Transformer 是由多个 ResidualAttentionBlock 堆叠而成的模块，用于提取序列特征。
    # 构建一个具有 layers 个残差注意力块的深度 Transformer 模型。
    transformer = Transformer(width=512, layers=6, heads=8)
    x = torch.randn(50, 1, 512)  # 50 个 token，512 维度
    output = transformer(x)
    '''
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    '''
    VisionTransformer 是 CLIP 模型中的图像编码器，用于将输入图像处理成适合 Transformer 的形式。
    划分图像为 Patch：通过卷积操作将输入图像分成多个小块（patch）。
    分类 Token (CLS)：增加一个特殊的标志，用于提取全局图像特征。
    位置编码：给每个 Patch 添加位置信息，帮助模型感知空间结构。
    特征提取：通过 Transformer 层提取图像的全局特征。
    '''
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # width相当于transform中的d_model
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # x=[1,3,224,224]
        x = self.conv1(x)  # shape = [*, width, grid, grid] # 将图片分成[32,32]个patch [1,768,7,7]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2],合并高宽 [1,768,49]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] ，更换位置 [1,49,768]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width],添加cls token[1,50,768]
        x = x + self.positional_embedding.to(x.dtype)  # 这里位置编码是可学习的参数，可能是切了path顺序让模型自己学习吧  [1,50,768]
        x = self.ln_pre(x)  # [1,50,768]

        x = x.permute(1, 0, 2)  # NLD -> LND  # [pixel,b,d_model]=[50,1,768]
        x = self.transformer(x)  # 多头transformer [50,1,768]
        x = x.permute(1, 0, 2)  # LND -> NLD  # [1,50,768]

        x = self.ln_post(x[:, 0, :])  # x[:, 0, :] 将所有信息汇聚到cls token中，只需前面来做下游任务 [1,768]

        if self.proj is not None:  # self.proj是可学习参数，维度为[768,512]
            x = x @ self.proj  # 通过学习参数将维度再次融合变成512特征，最终为[1,512]

        return x


class CLIP(nn.Module):
    '''
    CLIP 是完整的模型，将图像编码器（VisionTransformer）和文本编码器（Transformer）结合，用于跨模态对齐。

    参数：
        embed_dim：嵌入维度，即图像和文本的共同表示维度。
        image_resolution：输入图像的分辨率。
        vision_layers：视觉编码器的层数，可以是一个整数或一个元组。
        vision_width：视觉编码器的宽度。
        vision_patch_size：图像补丁的大小。
        context_length：文本上下文的最大长度。
        vocab_size：词汇表的大小。
        transformer_width：文本编码器的宽度。
        transformer_heads：文本编码器的多头注意力机制的头数。
        transformer_layers：文本编码器的层数。
    成员变量：
        self.context_length：文本上下文的最大长度。
        self.visual：视觉编码器，使用 VisionTransformer 实现。
        self.transformer：文本编码器，使用 Transformer 实现。
        self.vocab_size：词汇表的大小。
        self.token_embedding：词嵌入层，将词汇表中的词映射到向量。
        self.positional_embedding：位置嵌入层，用于添加位置信息。
        self.ln_final：最终的层归一化层。
        self.text_projection：文本投影矩阵，用于将文本编码器的输出投影到嵌入维度。
        self.logit_scale：对数尺度参数，用于调整对比损失的温度
    '''
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length  # 77

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  #
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # x 每个句子前面有值，有2个特殊符号[CLS]与[Seq]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]，[3,77,512]
        x = x + self.positional_embedding.type(self.dtype)  # 位置编码直接赋可学习位置，添加位置信息[3,77,512]
        x = x.permute(1, 0, 2)  # NLD -> LND,[77,3,512]
        x = self.transformer(x)  # 共11个 和图像encode结构一致 [77,3,512]
        x = x.permute(1, 0, 2)  # LND -> NLD，[3,77,512]
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # text.argmax(dim=-1) 句子最后有一个seq字段，是最大的，因此能获得句子个数数量
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features,# 每一行sqr(a1^2+a2^2+...)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)  # [batch_img,512]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)  # [batch_text,512]

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()  # 可学习参数
        logits_per_image = logit_scale * image_features @ text_features.t()  # 特征相乘获得相似度
        logits_per_text = logits_per_image.t()  # 变成文本

        # shape = [global_batch_size, global_batch_size]
        # print('图像模态的特征表示为：', image_features)
        # print('文本模态的特征表示为：', text_features)
        return logits_per_image, logits_per_text, image_features, text_features


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16
    在 CLIP 模型的代码中，convert_weights 方法是一个辅助函数，
    用于将模型的权重转换为特定的数据类型，通常是 float16（半精度浮点数）。
    这有助于减少模型的内存占用和加速推理过程，尤其是在 GPU 上运行时。
    """
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    '''
    build_model 方法的主要作用是根据配置参数创建并返回一个 CLIP 模型实例。它负责解析配置参数，并调用 CLIP 类的构造函数来创建模型。
    检测模型类型：
        检查 state_dict 中是否存在 visual.proj 键，以确定模型是否使用 ViT（Vision Transformer）。
        如果存在 visual.proj，则认为是 ViT 模型，否则是 ResNet 模型。
    解析 ViT 模型参数：
        vision_width：从 visual.conv1.weight 的形状中提取视觉编码器的宽度。
        vision_layers：统计 visual 层中包含 attn.in_proj_weight 的键的数量，以确定视觉编码器的层数。
        vision_patch_size：从 visual.conv1.weight 的形状中提取图像补丁的大小。
        grid_size：从 visual.positional_embedding 的形状中计算网格大小。
        image_resolution：根据 vision_patch_size 和 grid_size 计算图像分辨率。
    解析 ResNet 模型参数：
        counts：统计每个 visual.layer 中的子层数量，以确定视觉编码器的层数。
        vision_layers：将 counts 转换为元组。
        vision_width：从 visual.layer1.0.conv1.weight 的形状中提取视觉编码器的宽度。
        output_width：从 visual.attnpool.positional_embedding 的形状中计算输出宽度。
        image_resolution：根据 output_width 计算图像分辨率。
    解析通用参数：
        embed_dim：从 text_projection 的形状中提取嵌入维度。
        context_length：从 positional_embedding 的形状中提取文本上下文的最大长度。
        vocab_size：从 token_embedding.weight 的形状中提取词汇表的大小。
        transformer_width：从 ln_final.weight 的形状中提取文本编码器的宽度。
        transformer_heads：计算文本编码器的多头注意力机制的头数。
        transformer_layers：统计 transformer.resblocks 中的子层数量，以确定文本编码器的层数。
    创建 CLIP 模型实例：
        使用解析出的参数调用 CLIP 类的构造函数，创建 CLIP 模型实例。
    清理 state_dict：
        删除 state_dict 中不必要的键，如 input_resolution、context_length 和 vocab_size。
    转换权重：
        调用 convert_weights 方法将模型权重转换为 float16。
    加载权重：
        使用 model.load_state_dict(state_dict) 将权重加载到模型中。
    返回模型：
        返回构建好的 CLIP 模型实例。
    '''
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # 768
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])  # 12
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # 32
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # 7
        image_resolution = vision_patch_size * grid_size  # 32*7
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]  # 512
    context_length = state_dict["positional_embedding"].shape[0]  # 77
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # 49408
    transformer_width = state_dict["ln_final.weight"].shape[0]  # 512
    transformer_heads = transformer_width // 64  # 8
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))  # 12
    #               512            224              12            768           32
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)  # 构建模型
    #       77            49408           512                  8                  12

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


if __name__ == '__main__':
    embed_dim = 512
    image_resolution = 224
    vision_layers = 12
    vision_width = 768
    vision_patch_size = 32
    context_length = 77
    vocab_size = 49408
    transformer_width = 512
    transformer_heads = 8
    transformer_layers = 12
    #               512            224              12            768           32
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)  # 构建模型
    #       77            49408           512                  8                  12
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # preprocess =clip._transform
    preprocess = clip._transform(model.visual.input_resolution)

    image1 = preprocess(Image.open("./CLIP.png")).unsqueeze(0).to(device)
    image2 = preprocess(Image.open("./1711521985480.jpg")).unsqueeze(0).to(device)
    # 将两张图片堆叠成一个批次
    images = torch.cat([image1, image2], dim=0)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    print('输入图像形状：', images.shape)
    print('输入文本形状：', text.shape)

    with torch.no_grad():
        # 编码后图像特征尺寸
        image_features = model.encode_image(images)
        # 编码后文本特征尺寸
        text_features = model.encode_text(text)
        print('编码后图像形状：', image_features.shape)
        print('编码后文本形状：', text_features.shape)
        # logits_per_image 表示图像与文本之间的相似度 图像，[文本1，文本2，文本3]
        logits_per_image, logits_per_text = model(images, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

