import cv2 as cv
import numpy as np  

img = cv.imread("Lenna_(test_image).png")

cv.imshow("myimage", img)
cv.waitKey(0)

