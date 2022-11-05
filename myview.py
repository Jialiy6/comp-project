import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from Frame import Ui_Form
import qdarkstyle
import numpy as np
from utils import *
from models import SRResNet
import time
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
import scipy.misc
import scipy.signal
import scipy.ndimage
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
import scipy.ndimage
import scipy.special
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models import SRDataset
import matplotlib.pyplot as plt
import time
import tkinter
img_noise = 1
img_darken = 1

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 4  # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 评估参数
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

global original_niqe, test_niqe, time_cost


class MyClass(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyClass, self).__init__(parent)
        self.setupUi(self)
        self.SelectImg.clicked.connect(self.open_image)
        self.Nearest.clicked.connect(self.nearest)
        self.Bilinear.clicked.connect(self.bilinear)
        self.Bicnic.clicked.connect(self.bicubic)
        self.SRResNet.clicked.connect(self.srresnet)
        self.AddNoise.clicked.connect(self.gasuss_noise)
        self.modeleval.clicked.connect(self.model_eval)
        self.ImgDarken.clicked.connect(self.img_darken)

    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self,
                                                       "打开图片",
                                                       "",
                                                       " *.bmp;;*.png;;*.jpeg;;*.jpg;;All Files (*)")
        global img_moise
        img_moise = 1
        img = cv2.imread(imgName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = QImage(imgName)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(1)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.ShowImage.setScene(self.scene)
        self.ShowImage.show()
        global img_path
        img_path = imgName

    def gasuss_noise(self):

        # 添加高斯噪声
        # image: 原始图像
        # mean: 均值
        # var: 方差,越大，噪声越大
        self.welcome.setText("运行中...")
        QApplication.processEvents()
        mean = 0
        var = 0.001
        image = cv2.imread(img_path)
        image = np.array(image / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
        out = np.uint8(out * 255)  # 解除归一化
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(np.uint8(out))
        im.save('./results/add_noise.jpg')
        img = cv2.imread('./results/add_noise.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = QImage('./results/add_noise.jpg')
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.scene.addItem(self.item)
        # self.ShowImage.setScene(self.scene)
        self.ShowImage.show()
        global img_path_noise
        global img_noise
        img_noise = 2
        img_path_noise = './results/add_noise.jpg'
        self.welcome.setText("运行完成！")
        return noise

    def nearest(self):
        self.welcome.setText("运行中...")
        QApplication.processEvents()
        path = img_path
        if img_noise == 2:
            path = img_path_noise
        if img_darken == 2:
            path = img_path_darken
        img = cv2.imread(path)
        height, width, channels = img.shape
        new_height = height * scaling_factor
        new_width = width * scaling_factor
        emptyImage = np.zeros((new_height, new_width, channels), np.uint8)  # 返回来一个给定形状和类型的用0填充的数组
        # win32api.MessageBox(0, "运行中...", "提醒", win32con.MB_ICONWARNING)
        # messagebox.showinfo("提示", "我是一个提示框")
        start = time.time()
        for i in range(new_height):
            for j in range(new_width):
                x = int(i / scaling_factor)
                y = int(j / scaling_factor)
                emptyImage[i, j] = img[x, y]
        time_cost = '{:.3f} 秒'.format(time.time() - start)
        emptyImage2 = cv2.cvtColor(emptyImage, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(np.uint8(emptyImage2))
        im.save('./results/test_nearest.jpg')
        ref = np.array(Image.open(img_path).convert('LA'))[:, :, 0]  # ref
        dis = np.array(Image.open('./results/test_nearest.jpg').convert('LA'))[:, :, 0]  # dis
        original_niqe = '%0.3f' % niqe(ref)
        test_niqe = '%0.3f' % niqe(dis)
        self.time_cost.setText(time_cost)
        self.originalniqe.setText(original_niqe)
        self.testniqe.setText(test_niqe)
        cv2.imshow("nearest neighbor", emptyImage)
        self.welcome.setText("运行完成！")
        cv2.waitKey(0)

    def bilinear(self):
        self.welcome.setText("运行中...")
        QApplication.processEvents()
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        new_height = height * scaling_factor
        new_width = width * scaling_factor
        emptyImage = np.zeros((new_height, new_width, channels), np.uint8)
        value = [0, 0, 0]
        start = time.time()
        for i in range(new_height):
            for j in range(new_width):
                x = i / scaling_factor  # 在新图的坐标
                y = j / scaling_factor
                u = (i + 0.0) / scaling_factor - int(x)  # 小数部分 不加int的话没有意义
                v = (j + 0.0) / scaling_factor - int(y)
                x = int(x) - 1  # 不-1要越界
                y = int(y) - 1
                for k in range(3):
                    if x + 1 < new_height and y + 1 < new_width:
                        value[k] = int(img[x, y][k] * (1 - u) * (1 - v) +
                                       img[x, y + 1][k] * v * (1 - u) +
                                       img[x + 1, y][k] * (1 - v) * u +
                                       img[x + 1, y + 1][k] * u * v)  # 计算像素值
                emptyImage[i, j] = (value[0], value[1], value[2])
        time_cost = '{:.3f} 秒'.format(time.time() - start)
        self.time_cost.setText(time_cost)
        emptyImage2 = cv2.cvtColor(emptyImage, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(np.uint8(emptyImage2))
        im.save('./results/test_bilinear.jpg')
        ref = np.array(Image.open(img_path).convert('LA'))[:, :, 0]  # ref
        dis = np.array(Image.open('./results/test_bilinear.jpg').convert('LA'))[:, :, 0]  # dis
        original_niqe = '%0.3f' % niqe(ref)
        test_niqe = '%0.3f' % niqe(dis)
        self.time_cost.setText(time_cost)
        self.originalniqe.setText(original_niqe)
        self.testniqe.setText(test_niqe)
        cv2.imshow("Bilinear", emptyImage)
        self.welcome.setText("运行完成！")
        cv2.waitKey(0)

    def S(self, x):
        x = np.abs(x)
        if 0 <= x < 1:
            return 1 - 2 * x * x + x * x * x
        if 1 <= x < 2:
            return 4 - 8 * x + 5 * x * x - x * x * x
        else:
            return 0

    def bicubic(self):
        self.welcome.setText("运行中...")
        QApplication.processEvents()
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        new_height = height * scaling_factor
        new_width = width * scaling_factor
        emptyImage = np.zeros((new_height, new_width, channels), np.uint8)
        start = time.time()
        for i in range(new_height):
            for j in range(new_width):
                x = i / scaling_factor
                y = j / scaling_factor
                u = (i + 0.0) / scaling_factor - int(x)
                v = (j + 0.0) / scaling_factor - int(y)
                x = int(x) - 2  # 防止越界
                y = int(y) - 2

                if x >= new_height:  # 越界问题
                    new_height - 1
                if y >= new_width:
                    new_width - 1

                A = np.array([[self.S(1 + u), self.S(u), self.S(1 - u), self.S(2 - u)]])  # 四个横坐标的权重W(i)

                if (new_height) - 3 >= x >= 1 and (new_width) - 3 >= y >= 1:
                    B = np.array([[img[x - 1, y - 1], img[x - 1, y], img[x - 1, y + 1], img[x - 1, y + 2]],
                                  [img[x, y - 1], img[x, y], img[x, y + 1], img[x, y + 2]],
                                  [img[x + 1, y - 1], img[x + 1, y], img[x + 1, y + 1], img[x + 1, y + 2]],
                                  [img[x + 2, y - 1], img[x + 2, y], img[x + 2, y + 1], img[x + 2, y + 2]], ])
                    C = np.array([[self.S(1 + v)], [self.S(v)], [self.S(1 - v)], [self.S(2 - v)]])

                    blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]  # 计算乘积
                    green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]
                    red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]

                    def pixel(value):
                        if value > 255:
                            value = 255
                        elif value < 0:
                            value = 0
                        return value

                    blue = pixel(blue)
                    green = pixel(green)
                    red = pixel(red)
                    emptyImage[i, j] = np.array([blue, green, red], dtype=np.uint8)
        time_cost = '{:.3f} 秒'.format(time.time() - start)
        emptyImage2 = cv2.cvtColor(emptyImage, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(np.uint8(emptyImage2))
        im.save('./results/test_bicubic.jpg')
        ref = np.array(Image.open(img_path).convert('LA'))[:, :, 0]  # ref
        dis = np.array(Image.open('./results/test_bicubic.jpg').convert('LA'))[:, :, 0]  # dis
        original_niqe = '%0.3f' % niqe(ref)
        test_niqe = '%0.3f' % niqe(dis)
        self.time_cost.setText(time_cost)
        self.originalniqe.setText(original_niqe)
        self.testniqe.setText(test_niqe)
        cv2.imshow("cubic", emptyImage)
        self.welcome.setText("运行完成！")
        cv2.waitKey(0)

    def srresnet(self):
        self.welcome.setText("运行中...")
        QApplication.processEvents()
        # 预训练模型
        srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

        # 加载模型SRResNet
        checkpoint = torch.load(srresnet_checkpoint)
        srresnet = SRResNet(large_kernel_size=large_kernel_size,
                            small_kernel_size=small_kernel_size,
                            n_channels=n_channels,
                            n_blocks=n_blocks,
                            scaling_factor=scaling_factor)
        srresnet = srresnet.to(device)
        srresnet.load_state_dict(checkpoint['model'])

        srresnet.eval()
        model = srresnet

        # 加载图像
        img = Image.open(img_path, mode='r')
        img = img.convert('RGB')

        # 双线性上采样
        Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
        Bicubic_img.save('./results/test_bicubic.jpg')

        # 图像预处理
        lr_img = convert_image(img, source='pil', target='imagenet')
        lr_img.unsqueeze_(0)

        # 记录时间
        start = time.time()
        # 转移数据至设备
        lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet

        # 模型推理
        with torch.no_grad():
            sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
            sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
            sr_img.save('./results/test_srres.jpg')

        img = cv2.imread('./results/test_srres.jpg')
        time_cost = '{:.3f} 秒'.format(time.time() - start)
        ref = np.array(Image.open(img_path).convert('LA'))[:, :, 0]  # ref
        dis = np.array(Image.open('./results/test_srres.jpg').convert('LA'))[:, :, 0]  # dis
        original_niqe = '%0.3f' % niqe(ref)
        test_niqe = '%0.3f' % niqe(dis)
        cv2.imshow("srresnet", img)
        self.welcome.setText("运行完成！")
        self.time_cost.setText(time_cost)
        self.originalniqe.setText(original_niqe)
        self.testniqe.setText(test_niqe)
        cv2.waitKey(0)

    def model_eval(self):
        self.welcome.setText("运行中...")
        QApplication.processEvents()
        # 测试集目录
        data_folder = "./data/"
        global test_data_name
        # test_data_names = ["Set5", "Set14", "BSD100"]
        if self.SET5.isChecked():
            test_data_name = "Set5"
        elif self.SET14.isChecked():
            test_data_name = "Set14"
        elif self.BSD100.isChecked():
            test_data_name = "BSD100"
        # 预训练模型
        srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

        # 加载模型SRResNet
        checkpoint = torch.load(srresnet_checkpoint)
        srresnet = SRResNet(large_kernel_size=large_kernel_size,
                            small_kernel_size=small_kernel_size,
                            n_channels=n_channels,
                            n_blocks=n_blocks,
                            scaling_factor=scaling_factor)
        srresnet = srresnet.to(device)
        srresnet.load_state_dict(checkpoint['model'])

        srresnet.eval()
        model = srresnet

        # for test_data_name in test_data_names:

        # 数据加载器
        test_dataset = SRDataset(data_folder,
                                 kind='test',
                                 crop_size=0,
                                 scaling_factor=4,
                                 lr_type='imagenet',
                                 hr_type='[-1, 1]',
                                 test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                  pin_memory=True)

        # 记录每个样本 PSNR 和 SSIM值
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()

        # 记录测试时间
        start = time.time()

        with torch.no_grad():
            # 逐批样本进行推理计算
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # 数据移至默认设备
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

                # 前向传播
                sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

                # 计算 PSNR 和 SSIM
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                    0)  # (w, h), in y-channel
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                    0)  # (w, h), in y-channel
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                               data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                             data_range=255.)
                print(psnr)
                print(ssim)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

            # 输出平均PSNR和SSIM
            time_cost = '{:.3f} 秒'.format((time.time() - start) / len(test_dataset))
            imgPSNR = '{psnrs.avg:.3f}'.format(psnrs=PSNRs)
            imgSSIM = '{ssims.avg:.3f}'.format(ssims=SSIMs)
            self.lineEdit.setText(time_cost)
            self.PSNR.setText(imgPSNR)
            self.SSIM.setText(imgSSIM)
            self.welcome.setText("运行完成！")

    def img_darken(self):
        self.welcome.setText("运行中...")
        QApplication.processEvents()
        im1 = Image.open(img_path)
        im2 = im1.point(lambda p: p * 0.5)
        im2 = im2.convert('RGB')
        im2.save("./results/darken.jpg")
        global img_path_darken
        img_path_darken = './results/darken.jpg'
        global img_darken
        img_darken = 2
        self.welcome.setText("运行完成！")

    def show_imgs(self):
        img = Image.open("E:/woman_GT.bmp")
        img2 = Image.open("./results/add_noise.jpg")
        img3 = Image.open("./results/test_srres.jpg")
        # img4 = Image.open("./results/test_bicubic_cut.png")
        # img5 = Image.open("./results/test_srres.jpg")

        plt.figure(figsize=(60, 60))  # 设置窗口大小
        plt.subplot(2, 3, 1), plt.title('image')
        plt.imshow(img), plt.axis('off')
        plt.subplot(2, 3, 2), plt.title('add_noise')
        plt.imshow(img2), plt.axis('off')
        plt.subplot(2, 3, 3), plt.title('srresnet')
        plt.imshow(img3), plt.axis('off')
        # plt.subplot(2, 3, 4), plt.title('bicubic')
        # plt.imshow(img4), plt.axis('off')
        # plt.subplot(2, 3, 5), plt.title('srresnet')
        # plt.imshow(img5), plt.axis('off')
        plt.show()


def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))
    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm) ** 2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl) * (gam2 / gam1)  # *aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def ggd_features(imdata):
    nr_gam = 1 / prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq / E ** 2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                    alpha1, N1, bl1, br1,  # (V)
                    alpha2, N2, bl2, br2,  # (H)
                    alpha3, N3, bl3, bl3,  # (D1)
                    alpha4, N4, bl4, bl4,  # (D2)
                    ])


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)

    feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

    return feats


def niqe(inputImgData):

    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (
                patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (
                patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov + sample_cov) / 2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin = MyClass()
    myWin.show()
    sys.exit(app.exec_())


