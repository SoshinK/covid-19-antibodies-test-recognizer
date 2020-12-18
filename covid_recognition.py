from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import skimage.io as io
from utils import get_shpae, order_points, four_point_transform



def largest_contour_type1(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    _, bin_img = cv2.threshold(img_gray,200,255,cv2.THRESH_BINARY)
    
    bin_img = 255 - bin_img
    kernel = np.ones((5,5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11,11))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    
    bin_img = 255 - bin_img
        
    # plt.imshow(bin_img)
    # plt.show()

    contours, _ = cv2.findContours(np.uint8(bin_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key = cv2.contourArea)
    
    hull = cv2.convexHull(c)
    
    cntrs = cv2.drawContours(img.copy(), [hull], -1, (0,0,255), 2)
    # plt.imshow(cntrs)
    # plt.show()

    return hull


def classify_type_1(img):
    img_smooth = cv2.bilateralFilter(img, 20, 100, 200)

    contour = largest_contour_type1(img_smooth)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)


    img_cnt = cv2.drawContours(img.copy(), [box], -1, (0,0,255), 3)
    
    # plt.imshow(img_cnt)
    # plt.show()


    cropped = four_point_transform(img, box).astype(np.float32)
    
    cropped = cropped / [cropped[:, :, 0].mean(), cropped[:, :, 1].mean(), cropped[:, :, 2].mean()] * [127, 127, 127]
    cropped = np.uint8(cropped)

    window = cropped[int(cropped.shape[0] * 0.4) : int(cropped.shape[0] * 0.6), int(cropped.shape[1] * 0.4) : int(cropped.shape[1] * 0.6), 0]
    window = 255 - window

    # plt.imshow(window, cmap='gray')
    # plt.show()

    sums = []
    for i in range(window.shape[0]):
        sums.append(sum(window[i]))
        
    sums /= np.sum(sums)
    

    l = sums.shape[0]
    m = sums.mean()

    answer = []

    if max(sums[int(l / 8):int(3 * l / 8)] - m) > 0.0001:
        answer.append("C")
    if max(sums[int(3 * l / 8):int(5 * l / 8)] - m) >= 0.00009:
        answer.append("G")
    if max(sums[int(5 * l / 8):int(7 * l / 8)] - m) >= 0.00009:
        answer.append("M")

    return ' '.join(answer)

    
TRUNK_PATH = Path(__file__).parent


def test():
    print(TRUNK_PATH)
    for im_n in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]:
        img = cv2.imread(str(TRUNK_PATH / "dataset1" / "whole" / (str(im_n) + ".jpg")))
        # img = cv2.imread("covid_recognition/masha.jpg")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        answer = classify_type_1(img)
        print("Results:")
        print(im_n, ": ", answer)


if __name__ == "__main__":
    test()