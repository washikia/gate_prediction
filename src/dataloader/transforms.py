import os
from pathlib import Path
import glob
import cv2 as cv
from tools.helpers import get_gate_loc, add_label
import numpy as np
from PIL import Image

from torchvision.transforms import v2
from torchvision import tv_tensors



def mold_background_to_black(img):
    '''
    Converts the background of an image to black while preserving the mold regions.
    
    This function identifies mold regions in a grayscale image by thresholding,
    finds the contours of the mold, selects the two largest contours by perimeter,
    and sets everything outside these contours to black.
    
    Args:
        img: numpy.ndarray
            Input grayscale image (2D array with values 0-255)
    
    Returns:
        numpy.ndarray
            Modified grayscale image with background set to black (0) and 
            mold regions preserved. Same shape and dtype as input.
    
    Note:
        The threshold value (250) may need adjustment depending on your images.
        The function assumes white background (255) and darker mold regions.
    '''
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
        # Sort contours by arc length (perimeter) in descending order
        sorted_contours = sorted(contours, key=lambda c: cv.arcLength(c, closed=True), reverse=True)
        
        # Get the two contours with greatest length
        if len(sorted_contours) >= 2:
            two_largest_contours = sorted_contours[:2]
            # Draw both contours
            cv.drawContours(mask, two_largest_contours, -1, 255, thickness=cv.FILLED)
        else:
            # If only one contour, use it
            largest_contour = sorted_contours[0]
            cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)

    # Apply mask to image
    img[mask == 0] = 0  # set background to black

    return img


def transform_fixed_rotation(image: np.ndarray, labels: tv_tensors.KeyPoints, angle: int):
    '''
    Pads, rotates, and resizes image and labels to prevent mold from being cut.
    
    Args:
        image: Input grayscale image (numpy.ndarray)
        labels: KeyPoints tensor with gate coordinates
    
    Returns:
        tuple: (transformed_image, transformed_labels)
    '''
    img_lab_tuple = {
        "img": image,
        "labels": labels
    }

    transforms = v2.Compose([
        v2.Pad(15, fill=0),
        v2.RandomRotation(8),
        v2.Resize((256, 512))
    ])

    img_lab_tuple = transforms(img_lab_tuple)

    # After transforms, labels may no longer be a KeyPoints subclass.
    # Convert to a plain numpy array, then to a nested Python list for JSON.
    labels_arr = np.asarray(img_lab_tuple["labels"])

    return img_lab_tuple["img"], labels_arr.tolist()




def generate_transformed_dataset(input_root: str):
    '''
    This function generates the final dataset that will be used for training

    Args:
        input_root: the root of the dataset (D:\\washik_personal\\projects\\gate_prediction\\data\\processed)
    
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
            img_name = img_name.name    # keep this var: img_name

            # check if the image has a label
            label_path = input_root_path.parent / "labels" / "annotations.json"
            label = get_gate_loc(label_path, img_name)
            assert label is not None

            # save the grayscale image with same name without year
            img = mold_background_to_black(img)  # keep this var: img -> get other transforms from this
            save_path = output_path / img_name
            cv.imwrite(str(save_path), img)

            # fixed roation +10 of the image and labels
            labels_kp = tv_tensors.KeyPoints(data=label, canvas_size=(256,512))
            img_rotated_10, labels_rotated_10 = transform_fixed_rotation(img, labels_kp, 10)
            save_name = img_name + "_rotated_10.png"
            save_path_ = output_path / save_name
            cv.imwrite(str(save_path_), img_rotated_10)
            add_label(label_path, save_name, labels_rotated_10)

            
        print("passed")




if __name__ == "__main__":
    generate_transformed_dataset("D:\\washik_personal\\projects\\gate_prediction\\data\\toy")
