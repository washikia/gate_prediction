import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


Sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

Sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],  
                    [-1, -2, -1]])  



# pad image
def pad_image(image: np.ndarray, kernel_size: int) -> np.ndarray:
    '''
    This function pads the input image with zeros on all sides.
    It assumes that image dimensions are larger than kernel size
    '''
    pad = kernel_size // 2

    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')

    return padded_image



# Gaussian kernel
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    '''
    This takes the size of the kernel and standard deviation as input
    and returns a 2D Gaussian kernel.
    '''
    kernel = np.zeros((size, size), dtype=np.float32)
    centre = size // 2

    for y in range(-centre, centre + 1, 1):
        for x in range(-centre, centre + 1, 1):
            kernel[x+1, y+1] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x**2 + y**2) / (2 * sigma**2)))

    kernel /= np.sum(kernel)

    return kernel



# apply kernel
def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    '''
    This function applies a convolutional kernel to the input image.
    '''
    padded_image = pad_image(image, kernel.shape[0])
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y, x] = np.sum(kernel * padded_image[y:y+kernel.shape[0], x:x+kernel.shape[1]])

    return output



# apply Gaussian blur
def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    '''
    This function applies Gaussian blur to the input image.
    '''
    kernel = gaussian_kernel(kernel_size, sigma)
    return apply_kernel(image, kernel)



# apply Sobel filter
def apply_sobel_filter(image: np.ndarray) -> np.ndarray:
    '''
    This function applies the Sobel filter to the input image.
    '''
    grad_x = apply_kernel(image, Sobel_x)
    grad_y = apply_kernel(image, Sobel_y)
    edges = np.hypot(grad_x, grad_y)
    return grad_x, grad_y, edges



# Sobel direction vector
def sobel_direction(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    '''
    This function computes the direction of the Sobel gradients.
    '''
    direction = np.arctan2(grad_y, grad_x)
    return direction



# non max suppression
def non_max_suppression(edges: np.ndarray, direction: np.ndarray) -> np.ndarray:
    '''
    This function applies non-maximum suppression to the edges.
    '''
    Z = np.zeros_like(edges, dtype=np.float32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for y in range(1, edges.shape[0]-1):
        for x in range(1, edges.shape[1]-1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] <= 180):
                    q = edges[y, x + 1]
                    r = edges[y, x - 1]
                # angle 45
                elif (22.5 <= angle[y, x] < 67.5):
                    q = edges[y + 1, x - 1]
                    r = edges[y - 1, x + 1]
                # angle 90
                elif (67.5 <= angle[y, x] < 112.5):
                    q = edges[y + 1, x]
                    r = edges[y - 1, x]
                # angle 135
                elif (112.5 <= angle[y, x] < 157.5):
                    q = edges[y - 1, x - 1]
                    r = edges[y + 1, x + 1]

                if (edges[y, x] >= q) and (edges[y, x] >= r):
                    Z[y, x] = edges[y, x]
                else:
                    Z[y, x] = 0

            except IndexError as e:
                pass

    return Z



# hysteresis thresholding
def hysteresis_thresholding(image: np.ndarray, low_threshold: float=0.1, high_threshold: float=0.2) -> np.ndarray:
    '''
    This function applies hysteresis thresholding to the input image.
    '''
    high_threshold_value = image.max() * high_threshold
    low_threshold_value = high_threshold_value * low_threshold

    M, N = image.shape
    res = np.zeros((M, N), dtype=np.float32)

    strong = np.float32(255)
    weak = np.float32(75)

    strong_i, strong_j = np.where(image >= high_threshold_value)
    zeros_i, zeros_j = np.where(image < low_threshold_value)

    weak_i, weak_j = np.where((image <= high_threshold_value) & (image >= low_threshold_value))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    for y in range(1, M-1):
        for x in range(1, N-1):
            if (res[y, x] == weak):
                try:
                    if ((res[y + 1, x - 1] == strong) or (res[y + 1, x] == strong) or (res[y + 1, x + 1] == strong)
                        or (res[y, x - 1] == strong) or (res[y, x + 1] == strong)
                        or (res[y - 1, x - 1] == strong) or (res[y - 1, x] == strong) or (res[y - 1, x + 1] == strong)):
                        res[y, x] = strong
                    else:
                        res[y, x] = 0
                except IndexError as e:
                    pass

    return res



"""
# test  data\2025\25M1710D_front.png
img = Image.open("../../data/2025/25M1710D_front.png").convert("L")
img = np.array(img)
blurred = apply_gaussian_blur(img, 5, 1.0)
grad_x, grad_y, edges = apply_sobel_filter(blurred)
direction = sobel_direction(grad_x, grad_y)
# ...existing code...

nms_edges = non_max_suppression(edges, direction)
hyst_edges = hysteresis_thresholding(nms_edges)

plt.figure(figsize=(18, 10))

plt.subplot(3, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.title("Blurred Image")
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.title("Sobel X")
plt.imshow(grad_x, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.title("Sobel Y")
plt.imshow(grad_y, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.title("Edges")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.title("Direction Vector")
plt.imshow(direction, cmap='hsv')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.title("Edges (Canny)")
plt.imshow(nms_edges, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.title("Edges (Hysteresis)")
plt.imshow(hyst_edges, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.title("Edges (Hysteresis)")
plt.imshow(hyst_edges, cmap='hsv')
plt.axis('off')

plt.tight_layout()
plt.show()
# ...existing
"""