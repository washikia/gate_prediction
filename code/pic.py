import numpy as np
import cv2 as cv

img = cv.imread('25M1710D_front.png')
print(img.shape)

logo = cv.imread('logo.png')
print(logo.shape)

constant= cv.copyMakeBorder(logo,15,14,139,139,cv.BORDER_CONSTANT,value=[255,255,255])
print(constant.shape)
cv.imshow('constant', constant)

ret, mask = cv.threshold(constant, 50, 255, cv.THRESH_BINARY)
print(ret)
cv.imshow('mask', mask)

mask_inv = cv.bitwise_not(mask)
cv.imshow('mask_inv', mask_inv)

dst = cv.addWeighted(img, 0.7, constant, 0.3, 0)

cv.imshow('dst', dst)

mask_inv2 = cv.bitwise_not(dst)
cv.imshow('mask_inv2', mask_inv2)

cv.waitKey(0)
cv.destroyAllWindows()