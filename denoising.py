#*encoding=utf-8

import os
import cv2 as cv
import numpy as np
from multiprocessing import Process, Pool, cpu_count
from matplotlib import pyplot as plt


# 二值化
def threshold(dst, greyGap):
    # 阈值
    for row in range(len(dst)):
        for col in range(len(dst[row])):
            boolColor = dst[row][col] > greyGap
            boolColorUp = True
            boolColorDn = True
            boolColorLt = True
            boolColorRt = True
            if row > 0:
                boolColorUp = dst[row - 1][col] > greyGap
            if row < len(dst) - 1:
                boolColorDn = dst[row + 1][col] > greyGap
            if col > 0:
                boolColorLt = dst[row][col - 1] > greyGap
            if col < len(dst[row]) - 1:
                boolColorRt = dst[row][col + 1] > greyGap

            if boolColor and boolColorUp and boolColorDn and boolColorLt and boolColorRt:
                dst[row][col] = 255
    return dst


def UniqueDenoising(grey):
    print(len(grey))


def ImageDenoising(filepath, outputPath, level):
    pos = filepath.rfind("\\")
    imageName = filepath[pos+1:]
    print('———处理文件:' + imageName + '———')
    # 滤波强度(越高越强)
    h = [30, 40, 45]
    # 灰度阈值(越低越强)
    greyGap = [30, 25, 15]
    # print('读取文件中...')
    img = cv.imread(filepath)
    # print('转换灰度图中...')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print('OpenCV去噪中...')
    dst = cv.fastNlMeansDenoising(grey, None, h[level], 7, 21)
    # print('阈值二值化中...')
    result = threshold(dst, greyGap[level])
    cv.imwrite(outputPath + "\\" + imageName, result)
    print('———处理完成:' + imageName + '———')


def Main(images, outputPath, level):
    pool = Pool(processes=cpu_count()-2)
    for image in images:
        pool.apply_async(ImageDenoising, (image, outputPath, level))
    pool.close()
    pool.join()


if __name__ == '__main__':
    imgs = ["E:\\YT\\FW-0000001\\0001J.jpg"]
    Main(imgs, "E:\YT2", 0)
    # 单独调用测试
    # path = 'YT/FW-0000001/'
    # outputPath = 'After/FW-0000001/'
    # for i in range(1, 10):
    #     name = '000'+str(i)+'J.jpg'
    #     print('———处理文件:'+name+'———')
    #     result = ImageDenoising(path+name)
    #
    #     print('输出中...')
    #     cv.imwrite(outputPath+name, result)
    #     print('——————————————————————')

    # result = ImageDenoising('0025J.jpg')
    #
    # plt.subplot(121),plt.imshow(cv.imread(path+name), 'gray')
    # plt.subplot(122),plt.imshow(result, 'gray')
    # plt.show()