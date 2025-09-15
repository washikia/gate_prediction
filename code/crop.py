import cv2 as cv
import numpy as np

# crop using thresh holding
img = cv.imread('25M1710D_front.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

cv.imshow('Thresholded', thresh)

contour = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contour[0], -1, (0, 0, 128), 2)

cv.imshow('Contours', img)

edges = cv.Canny(img, 100, 200)
cv.imshow('Edges', edges)

x, y, w, h = cv.boundingRect(edges)
crop = img[y:y+h, x:x+w]
cv.imshow('Cropped', crop)

cv.waitKey(0)
cv.destroyAllWindows()

# cv.drawContours(img, contour[0], -1, (0, 0, 128), 2)