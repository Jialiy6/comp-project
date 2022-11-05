import cv2
import numpy as np
import math


def S(x):
    x = np.abs(x)
    if 0 <= x < 1:
        return 1 - 2 * x * x + x * x * x
    if 1 <= x < 2:
        return 4 - 8 * x + 5 * x * x - x * x * x
    else:
        return 0


def function(img, new_height, new_width):
    height, width, channels = img.shape
    emptyImage = np.zeros((new_height, new_width, channels), np.uint8)
    height_scale = new_height / height
    width_scale = new_width / width
    for i in range(new_height):
        for j in range(new_width):
            x = i / height_scale
            y = j / width_scale
            u = (i + 0.0) / height_scale - int(x)
            v = (j + 0.0) / width_scale - int(y)
            x = int(x) - 2  # 防止越界
            y = int(y) - 2

            if x >= new_height:  # 越界问题
                new_height - 1
            if y >= new_width:
                new_width - 1

            A = np.array([[S(1 + u), S(u), S(1 - u), S(2 - u)]])  # 四个横坐标的权重W(i)

            if (new_height - 3) >= x >= 1 and (new_width - 3) >= y >= 1:
                B = np.array([[img[x - 1, y - 1], img[x - 1, y], img[x - 1, y + 1], img[x - 1, y + 2]],
                              [img[x, y - 1], img[x, y], img[x, y + 1], img[x, y + 2]],
                              [img[x + 1, y - 1], img[x + 1, y], img[x + 1, y + 1], img[x + 1, y + 2]],
                              [img[x + 2, y - 1], img[x + 2, y], img[x + 2, y + 1], img[x + 2, y + 2]], ])
                C = np.array([[S(1 + v)], [S(v)], [S(1 - v)], [S(2 - v)]])

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

    return emptyImage


img = cv2.imread("e:\\test.jpg")
zoom = function(img, 520, 372)
cv2.imshow("cubic", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)