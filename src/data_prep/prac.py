from cv2.typing import MatLike


from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from torchvision.tv_tensors import KeyPoints
from torchvision.transforms import v2
# from helper import plot
import json


plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["savefig.bbox"] = "tight"

def get_gate_loc(data_path: str, image_path: str):
    '''
    get the base of the image from the image_path
    the data_path is a json file that contains the gate location for each image
    the coordinates are saved in dictionary format, return the value of the key that is the base of the image
    '''
    with open(data_path, 'r') as f:
        data = json.load(f)
    base = Path(image_path).name
    return data[base]



def mold_background_to_black(img_pil):
    # Convert to numpy array
    img_np = np.array(img_pil)

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Threshold: background white -> 255, mold -> 0
    # Adjust threshold depending on your images
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    print(contours[6])
    print(len(contours[6]))
    print(type(contours[6]))

    # Create empty mask
    mask = np.zeros_like(gray)

    # Fill the largest contour (assuming it's the mold)
    # if contours:
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)


if __name__ == "__main__":
    data_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\labels\\annotations.json'
    image_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\processed\\2025\\without_gate\\21M4870D_front.png'
    # gate_loc = get_gate_loc(data_path, image_path)
    # print(gate_loc)
    # plot([(Image.open(image_path), gate_loc)])
    ip = 'D:\\washik_personal\\projects\\gate_prediction\\data\\toy\\2021\\without_gate\\21M4870D_front.png'
    pil_img = Image.open(ip)
    mold_background_to_black(pil_img)