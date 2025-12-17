import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv
from random import choice, choices


def transform_point(point: list) -> list:
    '''
    Changes the coordiantes by 1 or 2 units in each direction.

    Args:
        list: of len 2, containing a single coordinate, like [a, b]

    Returns:
        list: the same format as the input but transformed
    '''
    
    sign = ["+", "-", None]
    weights = [0.45, 0.45, 0.1]
    new_coord = list    ()

    for coord in point:
        unit = choice([1,2])
        dir = choices(sign, weights= weights)[0]
        if dir == '+':
            new_coord.append(coord+unit)
        elif dir == '-':
            new_coord.append(coord-unit)
        else:
            new_coord.append(coord)
    
    return new_coord





def get_gate_loc(data_path: str, image_name: str):
    """
    Retrieve gate-location coordinates for a given image from a JSON label file.

    Args:
        data_path (str): Path to the JSON file containing image labels.
        image_name (str): Filename used as key in the JSON (e.g. "25M1710D_front.png").

    Returns:
        list | None: List of [x, y] coordinate pairs if present, otherwise None.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data.get(image_name, None)



def add_label(label_path, image_name, label_value):
    """Add or update a label entry for an image in the annotations JSON file."""
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Add / update entry
    data[image_name] = label_value  # e.g. label_value is a list of [x, y] points

    # 3. Save back to file
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



def save_tv_image(img_tensor, path):
    # Convert to numpy
    img_np = np.asarray(img_tensor, dtype=np.uint8)

    # For grayscale images stored as (1, H, W)
    if img_np.ndim == 3 and img_np.shape[0] == 1:
        img_np = img_np[0]

    cv.imwrite(str(path), img_np)



def show_images(img_list, titles=None, figsize=(15, 5)):
    """
    Display a list of images side by side.

    Args:
        img_list (list): List of images (PIL.Image or numpy arrays HxWxC or HxW).
        titles (list, optional): List of titles for each image.
        figsize (tuple): Size of the matplotlib figure.
    """
    n = len(img_list)
    plt.figure(figsize=figsize)
    
    for i, img in enumerate(img_list):
        plt.subplot(1, n, i + 1)
        
        # Convert PIL to numpy
        if isinstance(img, Image.Image):
            img_to_show = np.array(img)
        else:
            img_to_show = img
        
        # For grayscale images, set cmap
        if len(img_to_show.shape) == 2:
            plt.imshow(img_to_show, cmap='gray')
        else:
            plt.imshow(img_to_show)
        
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')
    
    plt.show()


# for i in range(10):
#     coord = [3, 4]
#     print(transform_point(coord))