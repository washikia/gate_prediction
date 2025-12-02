import os
from pathlib import Path
import glob
import cv2 as cv
from tools.helpers import get_gate_loc
import numpy as np
from PIL import Image



def mold_background_to_black(img):
    # Threshold: background white -> 255, mold -> 0
    # Adjust threshold depending on your images
    _, thresh = cv.threshold(img, 250, 255, cv.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        print(len(contour[0][0]), contour[0][0])

    # Create empty mask
    mask = np.zeros_like(img)

    # Fill the largest contour (assuming it's the mold)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)

    # Apply mask to image
    img[mask == 0] = 0  # set background to black

    return img



def generate_transformed_dataset(input_root: str):
    '''
    This function generates the final dataset that will be used for training

    Args:
        input_root: the root of the dataset (D:\washik_personal\projects\gate_prediction\data\processed)
    
    Returns:
        a Path object that points to the final dataset
    
    '''
    input_root_path = Path(input_root)
    output_path = input_root_path.parent / "final_dataset"
    output_path.mkdir(parents=True, exist_ok=True)

    for year in input_root_path.glob("*"):
        img_path = year / "without_gate"
        for img_name in img_path.glob("*.png"):
            img = cv.imread(str(img_name), cv.IMREAD_GRAYSCALE)
            img_name = img_name.name

            # check if the image has a label
            label_path = input_root_path.parent / "labels" / "annotations.json"
            assert get_gate_loc(label_path, img_name) is not None

            # save the grayscale image with same name without year
            img = mold_background_to_black(img)
            save_path = output_path / img_name
            cv.imwrite(str(save_path), img)


            
        print("passed")




if __name__ == "__main__":
    generate_transformed_dataset("D:\\washik_personal\\projects\\gate_prediction\\data\\toy")
