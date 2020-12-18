import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import skimage.io as io

from scipy.ndimage import gaussian_filter1d

def largest_contour_type1(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    _, bin_img = cv2.threshold(img_gray,200,255,cv2.THRESH_BINARY)
    
    # bin_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,91,10)

    # plt.imshow(bin_img)
    # plt.show()
    # plt.imshow(bin_img2)
    # plt.show()

    # bin_img = bin_img | bin_img2

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

def get_shpae(approx):
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        area = cv2.contourArea(approx)
        perimeter = cv2.arcLength(approx, True) 
        ar = area * 16 / perimeter**2
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    else:
        shape = "circle"
    return shape

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.reshape(4, 2).sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts.reshape(4, 2), axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


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

    



def test():
    for im_n in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]:
    # for im_n in [1]:
        img = cv2.imread("covid_recognition/dataset1/whole/" + str(im_n) + ".jpg")
        # img = cv2.imread("covid_recognition/masha.jpg")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        answer = classify_type_1(img)
        print("Results:")
        print(im_n, ": ", answer)


if __name__ == "__main__":
    test()