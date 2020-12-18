import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import skimage.io as io

from utils import get_shpae, order_points, four_point_transform


def largest_contour_type2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray,5)

    bin_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 12)
    
    bin_img = 255 - bin_img
    kernel = np.ones((5,5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11,11))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((5,5))
    bin_img = cv2.erode(bin_img, kernel, 1)
        
    contours, _ = cv2.findContours(np.uint8(bin_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key = cv2.contourArea)
    hull = c 
    
    return hull



def classify_type_2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_smooth = cv2.bilateralFilter(img, 20, 100, 200)
    contour = largest_contour(img_smooth)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cropped = four_point_transform(img, box).astype(np.float32)
    
    cropped = cropped / [cropped[:, :, 0].mean(), cropped[:, :, 1].mean(), cropped[:, :, 2].mean()] * [127, 127, 127]
    cropped = np.uint8(cropped)
    window = cropped[int(cropped.shape[0] * 0.35) : int(cropped.shape[0] * 0.5), int(cropped.shape[1] * 0.42) : int(cropped.shape[1] * 0.56), 0]
    window = 255 - window

    sums = []
    for i in range(window.shape[0]):
        sums.append(sum(window[i]))

    sums /= np.sum(sums)


    l = sums.shape[0]
    m = sums.mean()
    answer = []
    if max(sums[int(l / 8):int(3 * l / 8)] - m) > 0.0001:
        answer.append('C')
    if max(sums[int(3 * l / 8):int(5 * l / 8)] - m) >= 0.001:
        answer.append('G')
    if max(sums[int(5 * l / 8):int(7 * l / 8)] - m) >= 0.005:
        answer.append('M')
    if len(answer)==1 and answer[0] == 'C':
        answer.append(', no antibodies')
    return ' '.join(answer)