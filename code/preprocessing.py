from utils import canny_edge_detection
import cv2 as cv
from PIL import Image

img = Image.open("../data/2025/25M1750D_front.png").convert("L")
print(img.size)