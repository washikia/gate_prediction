import cv2 as cv
import numpy as np
from utils import canny_edge_detection

# crop using thresh holding
img = cv.imread('25M1710D_front.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

# cv.imshow('Thresholded', thresh)

# contour = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contour[0], -1, (0, 0, 128), 2)
# cv.imshow('Contours', img)

# edges = cv.Canny(img, 100, 200)
# cv.imshow('Edges', edges)

canny_edges = canny_edge_detection(gray, 3)
canny_edges = np.uint8(canny_edges)
cv.imshow('Canny Edges 1', canny_edges)

# canny_contour = cv.findContours(canny_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, canny_contour[0], -1, (0, 0, 128), 2)
# cv.imshow('Canny Contours', img)

x, y, w, h = cv.boundingRect(canny_edges)
crop = img[y:y+h, x:x+w]

print((crop.shape))
cv.imshow('Cropped', crop)


# --------------------------------------------------------------

img2 = cv.imread("../data/2025/25M1720D_front.png")
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
cv.imshow('Original', img2gray)

canny_edges = canny_edge_detection(img2gray, 3)
cv.imshow('Canny Edges 2', canny_edges)

canny_edges = np.uint8(canny_edges)
x, y, w, h = cv.boundingRect(canny_edges)
crop = img2[y:y+h, x:x+w]
cv.imshow('Cropped2', crop)

print((crop.shape))

cv.waitKey(0)
cv.destroyAllWindows()

# cv.drawContours(img, contour[0], -1, (0, 0, 128), 2)