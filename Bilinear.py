import cv2
import numpy as np
from PIL import Image
import math


def function(img, new_height, new_width):
    height, width, channels = img.shape
    print(height)
    emptyImage = np.zeros((new_height, new_width, channels), np.uint8)
    value = [0, 0, 0]
    height_scale = new_height / height
    width_scale = new_width / width
    print(height_scale)
    for i in range(new_height):
        for j in range(new_width):
            x = i / height_scale  # 在新图的坐标
            y = j / width_scale
            u = (i + 0.0) / height_scale - int(x)  # 小数部分 不加int的话没有意义
            v = float((j + 0.0) / width_scale) - int(y)
            x = int(x) - 1  # 不-1要越界
            y = int(y) - 1
            for k in range(3):
                if x + 1 < new_height and y + 1 < new_width:
                    value[k] = int(img[x, y][k] * (1 - u) * (1 - v) +
                                   img[x, y + 1][k] * v * (1 - u) +
                                   img[x + 1, y][k] * (1 - v) * u +
                                   img[x + 1, y + 1][k] * u * v)  # 计算像素值
            emptyImage[i, j] = (value[0], value[1], value[2])
    return emptyImage


def gasuss_noise(img, mean=0, var=0.001):

    # 添加高斯噪声
    # image:原始图像
    # mean : 均值
    # var : 方差,越大，噪声越大

    # image = cv2.imread(img_path)
    img = np.array(img / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var ** 0.5, img.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = img + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复
    cv2.imshow("gasuss", out)
    # out.save('./results/add_noise.jpg')
    noise = noise * 255
    return out

img = cv2.imread("e:\\test.jpg")
# zoom = function(img, 2048, 2048)

out = gasuss_noise(img, mean=0, var=0.003)
im = Image.fromarray(out)
im.save('./results/add_noise.jpg')
cv2.imshow("Bilinear Interpolation", out)
cv2.imshow("image", img)
cv2.waitKey(0)