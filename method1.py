import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Import the image
for i in range(10, 31):
    img = cv.imread('FW-0000001/00'+str(i)+'J.jpg')
    # dst = cv.fastNlMeansDenoising(img,None,50,7,21)
    dst = cv.fastNlMeansDenoisingColored(img,None,50,50,7,21)
    grey = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    gauss = cv.GaussianBlur(grey, (1, 1), 0)
    result = cv.adaptiveThreshold(gauss, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=5, C=4)

    dst2 = cv.fastNlMeansDenoising(result,None,50,7,21)
    result2 = cv.adaptiveThreshold(dst2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=5, C=4)
# dst3 = cv.fastNlMeansDenoising(result2,None,50,7,21)
# result3 = cv.adaptiveThreshold(dst3, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=5, C=4)
# dst4 = cv.fastNlMeansDenoising(result3,None,50,7,21)
# result4 = cv.adaptiveThreshold(dst4, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=5, C=4)
#
# dst5 = cv.fastNlMeansDenoising(result2,None,50,7,21)
# # gauss2 = cv.GaussianBlur(dst5, (1, 1), 0)
# result5 = cv.adaptiveThreshold(dst5, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=5, C=4)

    cv.imwrite('FW/00'+str(i)+'J.jpg', result2)
    print(str(i)+' finished')
# plt.subplot(121),plt.imshow(img, 'gray')
# plt.subplot(122),plt.imshow(result2, 'gray')
# plt.show()

# cv.imshow('image', gauss)
# cv.waitKey(0)
# cv.destroyAllWindows()