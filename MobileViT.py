import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 卷积层的辅助函数
def create_conv_layer(input_channels, output_channels, kernel_size=3, stride=1, padding=0, groups=1, bias=False, normalization=True, activation=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias, groups=groups))
    
    # 如果需要批归一化
    if normalization:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(output_channels))
    
    # 如果需要激活函数
    if activation:
        conv.add_module('Activation', nn.SiLU())
    
    return conv

# 倒残差块
class InvertedResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]  # 步幅只能是1或2
        hidden_dimension = int(round(input_channels * expand_ratio))  # 扩展后的维度
        self.block = nn.Sequential()
        
        # 如果扩展比率不为1，进行1x1卷积扩展
        if expand_ratio != 1:
            self.block.add_module('expansion_1x1', create_conv_layer(input_channels, hidden_dimension, kernel_size=1, stride=1, padding=0))
        
        # 进行3x3深度可分离卷积
        self.block.add_module('conv_3x3', create_conv_layer(hidden_dimension, hidden_dimension, kernel_size=3, stride=stride, padding=1, groups=hidden_dimension))
        
        # 进行1x1卷积压缩
        self.block.add_module('reduction_1x1', create_conv_layer(hidden_dimension, output_channels, kernel_size=1, stride=1, padding=0, activation=False))
        
        # 如果步幅为1且输入输出通道相同，则使用残差连接
        self.use_residual_connection = self.stride == 1 and input_channels == output_channels

    def forward(self, x):
        if self.use_residual_connection:
            return x + self.block(x)  # 使用残差连接
        else:
            return self.block(x)  # 不使用残差连接，直接返回

# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension, num_heads=4, dimension_per_head=8, attention_dropout=0):
        super().__init__()
        self.qkv_projection = nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=True)  # qkv投影
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension)  # 输出投影
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.scale_factor = dimension_per_head ** -0.5  # 缩放因子

    def forward(self, x):
        batch_size, sequence_length, input_channels = x.shape
        # 对输入进行qkv投影，并按头分开
        qkv = self.qkv_projection(x).reshape(batch_size, sequence_length, 3, self.num_heads, -1)
        qkv = qkv.transpose(1, 3).contiguous()  # 转置后连续存储
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # 获取q, k, v

        q = q * self.scale_factor  # 缩放q
        k = k.transpose(-1, -2)  # 转置k

        # 计算注意力分数
        attention_scores = torch.matmul(q, k)
        attention_scores_float = self.softmax(attention_scores.float())  # 计算softmax
        attention_scores = attention_scores_float.to(attention_scores.dtype)  # 恢复为原来的dtype
        attention_scores = self.attention_dropout(attention_scores)  # 应用注意力掉落

        # 使用注意力分数计算输出
        attention_output = torch.matmul(attention_scores, v)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        attention_output = self.output_projection(attention_output)  # 输出投影

        return attention_output

# Transformer模块
class Transformer(nn.Module):
    def __init__(self, embedding_dimension, feedforward_latent_dimension, num_heads=8, dimension_per_head=8, dropout=0, attention_dropout=0):
        super().__init__()
        self.pre_norm_multihead_attention = nn.Sequential(
            nn.LayerNorm(embedding_dimension, eps=1e-5, elementwise_affine=True),  # 层归一化
            MultiHeadAttention(embedding_dimension, num_heads, dimension_per_head, attention_dropout),  # 多头自注意力
            nn.Dropout(dropout)
        )
        
        self.pre_norm_feedforward_network = nn.Sequential(
            nn.LayerNorm(embedding_dimension, eps=1e-5, elementwise_affine=True),  # 层归一化
            nn.Linear(embedding_dimension, feedforward_latent_dimension, bias=True),  # 全连接层
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_latent_dimension, embedding_dimension, bias=True),  # 输出层
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.pre_norm_multihead_attention(x)  # 经过多头自注意力
        x = x + self.pre_norm_feedforward_network(x)  # 经过前馈网络
        return x

# MobileViTBlock模块
class MobileViTBlock(nn.Module):
    def __init__(self, input_channels, attention_dimension, feedforward_multiplier, num_heads, dimension_per_head, num_attention_blocks, patch_size):
        super(MobileViTBlock, self).__init__()
        self.patch_height, self.patch_width = patch_size
        self.patch_area = int(self.patch_height * self.patch_width)  # 计算每个patch的面积

        # 局部特征表示部分
        self.local_representation = nn.Sequential()
        self.local_representation.add_module('conv_3x3', create_conv_layer(input_channels, input_channels, kernel_size=3, stride=1, padding=1))
        self.local_representation.add_module('conv_1x1', create_conv_layer(input_channels, attention_dimension, kernel_size=1, stride=1, normalization=False, activation=False))

        # 全局特征表示部分（基于Transformer）
        self.global_representation = nn.Sequential()
        feedforward_dimensions = [int((feedforward_multiplier * attention_dimension) // 16 * 16)] * num_attention_blocks  # 前馈维度
        for i in range(num_attention_blocks):
            feedforward_dimension = feedforward_dimensions[i]
            self.global_representation.add_module(f'Transformer_{i}', Transformer(attention_dimension, feedforward_dimension, num_heads, dimension_per_head))  # Transformer模块
        self.global_representation.add_module('LayerNorm', nn.LayerNorm(attention_dimension, eps=1e-5, elementwise_affine=True))  # 最后的LayerNorm

        # 卷积投影和特征融合
        self.convolution_projection = create_conv_layer(attention_dimension, input_channels, kernel_size=1, stride=1)
        self.fusion = create_conv_layer(2 * input_channels, input_channels, kernel_size=3, stride=1)

    # 特征展开，将特征图分解成patches
    def unfolding(self, feature_map):
        patch_width, patch_height = self.patch_width, self.patch_height
        batch_size, in_channels, original_height, original_width = feature_map.shape

        # 计算需要的新的高度和宽度
        new_height = int(math.ceil(original_height / self.patch_height) * self.patch_height)
        new_width = int(math.ceil(original_width / self.patch_width) * self.patch_width)

        # 如果图像尺寸不匹配，进行插值调整
        interpolate = False
        if new_width != original_width or new_height != original_height:
            feature_map = F.interpolate(
                feature_map, size=(new_height, new_width), mode="bilinear", align_corners=False
            )
            interpolate = True

        # 计算patches的数量
        num_patches_width = new_width // patch_width
        num_patches_height = new_height // patch_height
        total_patches = num_patches_height * num_patches_width

        reshaped_feature_map = feature_map.reshape(
            batch_size * in_channels * num_patches_height, patch_height, num_patches_width, patch_width
        )
        transposed_feature_map = reshaped_feature_map.transpose(1, 2)
        reshaped_feature_map = transposed_feature_map.reshape(
            batch_size, in_channels, total_patches, self.patch_area
        )
        transposed_feature_map = reshaped_feature_map.transpose(1, 3)
        patches = transposed_feature_map.reshape(batch_size * self.patch_area, total_patches, -1)

        info_dict = {
            "original_size": (original_height, original_width),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": total_patches,
            "num_patches_width": num_patches_width,
            "num_patches_height": num_patches_height,
        }

        return patches, info_dict

    # 特征折叠，将patches恢复为特征图
    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)

        patches = patches.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = patches.size()
        num_patches_height = info_dict["num_patches_height"]
        num_patches_width = info_dict["num_patches_width"]

        patches = patches.transpose(1, 3)

        feature_map = patches.reshape(
            batch_size * channels * num_patches_height, num_patches_width, self.patch_height, self.patch_width
        )
        feature_map = feature_map.transpose(1, 2)
        feature_map = feature_map.reshape(
            batch_size, channels, num_patches_height * self.patch_height, num_patches_width * self.patch_width
        )
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["original_size"],
                mode="bilinear",
                align_corners=False,
            )
        return feature_map

    def forward(self, x):
        residual = x.clone()
        x = self.local_representation(x)  # 局部特征表示
        x, info_dict = self.unfolding(x)  # 展开成patches
        x = self.global_representation(x)  # 全局表示（Transformer模块）
        x = self.folding(x, info_dict)  # 恢复为特征图
        x = self.convolution_projection(x)  # 卷积投影
        x = self.fusion(torch.cat((residual, x), dim=1))  # 特征融合
        return x

# MobileViT模型
class MobileViT(nn.Module):
    def __init__(self, image_size, mode, num_classes, patch_size=(2, 2)):
        super().__init__()
        image_height, image_width = image_size
        self.patch_height, self.patch_width = patch_size
        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0  # 图像尺寸必须是patch尺寸的整数倍
        assert mode in ['xx_small', 'x_small', 'small'] # 三种模式

        if mode == 'xx_small':
            mobilevit_expansion_multiplier = 2
            feedforward_multiplier = 2
            last_layer_expansion_factor = 4
            channels = [16, 16, 24, 48, 64, 80]
            attention_dimension = [64, 80, 96]
        elif mode == 'x_small':
            mobilevit_expansion_multiplier = 4
            feedforward_multiplier = 2
            last_layer_expansion_factor = 4
            channels = [16, 32, 48, 64, 80, 96]
            attention_dimension = [96, 120, 144]
        elif mode == 'small':
            mobilevit_expansion_multiplier = 4
            feedforward_multiplier = 2
            last_layer_expansion_factor = 4
            channels = [16, 32, 64, 96, 128, 160]
            attention_dimension = [144, 192, 240]
        else:
            raise NotImplementedError

        self.conv_initial = create_conv_layer(3, channels[0], kernel_size=3, stride=2)

        self.layer_1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mobilevit_expansion_multiplier)
        )
        self.layer_2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mobilevit_expansion_multiplier),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mobilevit_expansion_multiplier),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mobilevit_expansion_multiplier)
        )
        self.layer_3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mobilevit_expansion_multiplier),
            MobileViTBlock(channels[3], attention_dimension[0], feedforward_multiplier, num_heads=4, dimension_per_head=8, num_attention_blocks=2, patch_size=patch_size)
        )
        self.layer_4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mobilevit_expansion_multiplier),
            MobileViTBlock(channels[4], attention_dimension[1], feedforward_multiplier, num_heads=4, dimension_per_head=8, num_attention_blocks=4, patch_size=patch_size)
        )
        self.layer_5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mobilevit_expansion_multiplier),
            MobileViTBlock(channels[5], attention_dimension[2], feedforward_multiplier, num_heads=4, dimension_per_head=8, num_attention_blocks=3, patch_size=patch_size)
        )
        self.conv_1x1_expansion = create_conv_layer(channels[-1], channels[-1] * last_layer_expansion_factor, kernel_size=1, stride=1)
        self.output_layer = nn.Linear(channels[-1] * last_layer_expansion_factor, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_initial(x)
        x = self.layer_1(x)
        x = self.layer_2(x) 
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_expansion(x)
        
        # 对最终输出进行全局平均池化
        x = torch.mean(x, dim=[-2, -1])
        x = self.output_layer(x)

        return x

# 测试模型
if __name__ == '__main__':
    model = MobileViT(image_size=(224, 224), mode='xx_small', num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.size())