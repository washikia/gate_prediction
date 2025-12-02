import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_gate_loc(data_path: str, image_name: str):
    '''
    get the base of the image from the image_path
    the data_path is a json file that contains the gate location for each image
    the coordinates are saved in dictionary format, return the value of the key that is the base of the image
    '''
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data.get(image_name, None)


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