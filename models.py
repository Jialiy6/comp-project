import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import ImageTransforms
from torch import nn
import math


class SRDataset(Dataset):

    # 数据集加载器

    def __init__(self, path, kind, crop_size, scaling_factor, lr_type, hr_type, test_data_name=None):

        # path: Json数据文件所在文件夹路径
        # kind: 'train' 或者 'test'
        # crop_size: 高分辨率图像裁剪尺寸 截取原图的一个子块放大
        # scaling_factor: 放大比例
        # lr_img_type: 低分辨率图像预处理方式
        # hr_img_type: 高分辨率图像预处理方式
        # test_data_name: 待评估数据集名称

        self.path = path
        self.kind = kind.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_type = lr_type
        self.hr_type = hr_type
        self.test_data_name = test_data_name

        assert self.kind in {'train', 'test'}
        if self.kind == 'test' and self.test_data_name is None:
            raise ValueError("请提供测试数据集名称!")
        assert lr_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet'}
        assert hr_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet'}

        # 如果是训练，则所有图像必须保持固定的分辨率以此保证能够整除放大比例
        if self.kind == 'train':
            assert self.crop_size % self.scaling_factor == 0, "裁剪尺寸不能被放大比例整除!"

        # 读取图像路径
        if self.kind == 'train':
            with open(os.path.join(path, 'train_images.json'), 'r') as j:
                self.images = json.load(j)
        else:
            with open(os.path.join(path, self.test_data_name + '_test_images.json'), 'r') as j:
                self.images = json.load(j)

        # 数据处理方式
        self.transform = ImageTransforms(kind=self.kind,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_type=self.lr_type,
                                         hr_type=self.hr_type)

    def __getitem__(self, i):

        # i: 图像检索号
        # 返回: 返回第i个低分辨率和高分辨率的图像对
        # 读取图像
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        if img.width <= 96 or img.height <= 96:
            print(self.images[i], img.width, img.height)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):

        # 返回: 加载的图像总数

        return len(self.images)


class ConvolutionalBlock(nn.Module):

    # 卷积模块：卷积层 BN归一化层（不用） 激活层

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):

        # in_channels: 输入通道数 RGB大多为3通道
        # out_channels: 输出通道数 每个卷积层中卷积核的数量
        # kernel_size: 核大小
        # stride: 步长
        # batch_norm: 是否包含BN层
        # activation: 激活层类型

        super(ConvolutionalBlock, self).__init__()

        # 激活函数
        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层
        # 输入为 in_channels = 64 张图像，输出 out_channels = 64 张特征图，卷积核为 3 * 3 的正方形
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=kernel_size // 2))

        # 1个BN归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):

        # 前向传播

        # input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        # 返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)

        output = self.conv_block(input)

        return output


class SubPixelConvolutionalBlock(nn.Module):

    # 子像素卷积模块:卷积 像素清洗 激活层
    # 特征图放大

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):

        # kernel_size: 卷积核大小
        # n_channels: 输入和输出通道数
        # scaling_factor: 放大比例

        super(SubPixelConvolutionalBlock, self).__init__()

        # 首先通过卷积将通道数扩展为 scaling factor^2 倍
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # 进行像素清洗，合并相关通道数据 上采样
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        # 最后添加激活层
        self.prelu = nn.PReLU()

    def forward(self, input):

        # 前向传播

        # input: 输入图像数据集，张量表示，大小为(N, n_channels, w, h)
        # 返回: 输出图像数据集，张量表示，大小为 (N, n_channels, w * scaling factor, h * scaling factor)

        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):

    # 残差模块：两个卷积模块 一个跳连

    def __init__(self, kernel_size=3, n_channels=64):

        # kernel_size: 核大小
        # n_channels: 输入和输出通道数（由于是ResNet网络，需要做跳连，因此输入和输出通道数是一致的）

        super(ResidualBlock, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):

        # 前向传播

        # input: 输入图像集，张量表示，大小为 (N, n_channels, w, h)
        # 返回: 输出图像集，张量表示，大小为 (N, n_channels, w, h)

        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class SRResNet(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):

        # large_kernel_size: 第一层卷积和最后一层卷积核大小
        # small_kernel_size: 中间层卷积核大小
        # n_channels: 中间层通道数
        # n_blocks: 残差模块数
        # scaling_factor: 放大比例

        # 复制并使用SRResNet的父类的初始化方法，即先运行nn.Module的初始化函数
        super(SRResNet, self).__init__()
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "放大比例必须为 2、 4 或 8!"

        # 卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels,
                                              kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # 卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # 子像素卷积模块, 每个模块放大两倍
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # 卷积块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        # 前向传播
        # lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        # 返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)

        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        residual = output  # (16, 64, 24, 24)
        output = self.residual_blocks(output)  # (16, 64, 24, 24)
        output = self.conv_block2(output)  # (16, 64, 24, 24)
        output = output + residual  # (16, 64, 24, 24)
        output = self.subpixel_convolutional_blocks(output)  # (16, 64, 24 * 4, 24 * 4)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)

        return sr_imgs
