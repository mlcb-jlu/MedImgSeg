import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import nibabel as nib
import os
import cv2
import math
# CVPR论文最终用的这个版本的代码把mask转为二值标签

def water(img_path):
    src = cv2.imread(img_path)
    img = src.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 消除噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 膨胀
    kernel2 = np.ones((7, 7), np.uint8)
    sure_bg = cv2.dilate(opening, kernel2, iterations=3)

    # 距离变换
    dist_transform = cv2.distanceTransform(sure_bg, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 获得未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记
    ret, markers1 = cv2.connectedComponents(sure_fg)

    # 确保背景是1不是0
    markers = markers1 + 1

    # 未知区域标记为0
    markers[unknown == 255] = 0

    markers3 = cv2.watershed(img, markers)
    # -1是分割线
    img[markers3 == -1] = [0, 0, 0]
    #1是背景
    img[markers3 == 1] = [0, 0, 0]
    #234是前景
    img[markers3 == 2] = [255, 255, 255]
    img[markers3 == 3] = [255, 255, 255]
    img[markers3 == 4] = [255, 255, 255]
    return img

def regional_growth(img_path):
    im = Image.open(img_path)  # 读取图片
    # im.show()

    im_array = np.array(im)
    im_array = im_array[:, :, 0]
    re = np.argwhere(im_array <= np.min(im_array)+20)

    dis_list = []
    for i in range(len(re)):
        ax, ay = re[i]
        dis = math.sqrt((abs(127-ax) ** 2) + (abs(127-ay) ** 2))
        dis_list.append(dis)
        a_i = np.argmin(dis_list)

    ax, ay = re[a_i]
    # print(im_array)
    [m, n] = im_array.shape

    a = np.zeros((m, n))  # 建立等大小空矩阵
    a[127, 127] = 1  # 设立种子点
    k = 20  # 设立区域判断生长阈值

    m = m-2
    n = n-2

    flag = 1  # 设立是否判断的小红旗
    while flag == 1:
        flag = 0
        lim = (np.cumsum(im_array * a)[-1]) / (np.cumsum(a)[-1])
        for i in range(2, m):
            for j in range(2, n):
                if a[i, j] == 1:
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if a[i + x, j + y] == 0:
                                if (abs(im_array[i + x, j + y] - lim) <= k):
                                    flag = 1
                                    a[i + x, j + y] = 1

    data = 255 * a  # 矩阵相乘获取生长图像的矩阵
    return data

def segmentation(img_path):
    src= cv2.imread(img_path)
    img = src.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # gray = cv2.convertScaleAbs(gray, alpha=0.7, beta=0)
    #中值滤波
    # gray = cv2.medianBlur(gray, 3)

    #二值分割
    # ret, thresh = cv2.threshold(gray, 180, 0, cv2.THRESH_TOZERO)
    # ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, gray.max(), 255,  cv2.THRESH_OTSU)
    #cv2.imshow('thresh', thresh)

    #开闭运算
    kernel = np.ones((3, 3), np.uint8)
    # 闭运算 = 先膨胀运算，再腐蚀运算（看上去将两个细微连接的图块封闭在一起）
    #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('closing', closing)


    #开运算 = 先腐蚀运算，再膨胀运算（看上去把细微连在一起的两块目标分开了）
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('opening', opening)

    dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    #cv2.imshow('dilate', dilate)

    # medianBlur = cv2.medianBlur(opening, 3)



    return opening


    # source_path = r"/home/dw/Disk_8T/ICCV2021"
    #
    # path = source_path + "/masks"
    #
    # output_path = source_path + '/labels'
def findCont(model_name, path, output_path):

    # path = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/results/selfie2anime/results_mask_reverse" + model_name
    # path = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/results/selfie2anime/results_mask_reverse" + model_name + 'teacher'

    # output_path = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/iccv_labels" + '/' + 'iccv_labels' + model_name
    # output_path = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/iccv_labels" + '/' + 'iccv_labels' + model_name + 'teacher'


    path_list = os.listdir(path)
    path_list.sort()
    len1 = 0

    # black = np.zeros((256,256))
    # cv2.imshow('black',black)
    # cv2.waitKey(0)
    count = 0
    for filename in path_list:
        count += 1
        cont_area = []
        len1 += 1
        image_path = os.path.join(path, filename)
        src = cv2.imread(image_path)
        result = water(image_path)
        # result = segmentation(image_path)

        # contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # #轮廓数量：
        # #print(len(contours))
        # #计算轮廓面积：
        # for i in range(len(contours)):
        #     cont = cv2.contourArea(contours[i])
        #     cont_area.append(cont)
        # #print(cont_area)
        # cont_area_ = cont_area
        # #-1防止只有两个轮廓时返回索引错误
        # # cont_area_[np.argmax(cont_area_)] = np.min(cont_area_) - 1
        # # cont_area_[np.argmax(cont_area_)] = np.min(cont_area_) - 1
        # # if np.max(cont_area_)>30000 or np.max(cont_area_)<400:
        # #     continue
        # #print("area中第二大的数为{}，位于第{}位".format(np.max(cont_area_), np.argmax(cont_area_) + 1))
        # print(filename+ "病灶区域面积为{}".format(np.max(cont_area_)))

        # black = np.zeros((256, 256))
        # #cv2.drawContours函数（原图，轮廓，轮廓索引，线条颜色，线条粗细）线条粗细为负数时填充轮廓内部
        # img = cv2.drawContours(src, contours, np.argmax(cont_area_), (0, 0, 255), 3)
        # # img = cv2.drawContours(black, contours, np.argmax(cont_area_), (255, 255, 255), -1)

        # cv2.imshow('drawimg', img)
        # cv2.waitKey(0)

        index = filename.rfind('.')
        filename = filename[:index]
        # if result[2, 2] == 255:
        #     result = 255-result
        filename = filename[:-5]+'_segmentation'

        cv2.imwrite(output_path + '/' + filename+".png", result)
        print(round(count * 100 / len(path_list), 2), "%")
        #cv2.imwrite(r'C:\Users\SY\Desktop\results\pair/' + filename + ".png", src)
        #进度：
        #print(len1 * 100 / len(path_list), "%")