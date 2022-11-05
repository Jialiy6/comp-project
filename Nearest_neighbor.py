import cv2
import numpy as np


def Nearest(img_path, new_height, new_width):
    print(img_path)
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    emptyImage = np.zeros((new_height, new_width, channels), np.uint8)  # 返回来一个给定形状和类型的用0填充的数组
    for i in range(new_height):
        for j in range(new_width):
            x = int(i / (new_height/height))
            y = int(j / (new_width/width))
            emptyImage[i, j] = img[x, y]
    cv2.imshow("nearest neighbor", emptyImage)
    cv2.waitKey(0)


# img = cv2.imread(openimage())
# zoom = function("E:/test.jpg", 1024, 1024)
# cv2.imshow("nearest neighbor", zoom)
# cv2.imshow("image", img)
# cv2.waitKey(0)