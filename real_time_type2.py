import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import skimage.io as io

from recognition_type2 import largest_contour_type2, classify_type_2

def real_time_test():
    cap = cv2.VideoCapture(0); 
    while(1): 
        # read frames 
        ret, img = cap.read(); 
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img_smoothed = cv2.bilateralFilter(img, 20, 100, 200)
        try:
            contour = largest_contour_type2(img)
        except ValueError:
            print('Oops')
            continue
        if len(contour) == 0:
            continue

        img = cv2.drawContours(img, [contour], -1, (0, 0, 255), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # org 
        org = (50, 50) 

        # fontScale :
        fontScale = 1

        # Blue color in BGR 
        color = (255, 0, 0) 

        # Line thickness of 2 px 
        thickness = 2

        text = ""
        # Using cv2.putText() method 
        try:
            text = classify_type_2(img)
        except:
            print('Oops')
        
        img = cv2.putText(img, text, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Original', img); 
        k = cv2.waitKey(30) & 0xff; 
        if k == 27: 
            break




if __name__ == "__main__":
    real_time_test()