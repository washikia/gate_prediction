# go through the data/raw folder and apply resize_and_pad to each image, saving to data/processed
# also maintain folder structure - the processed images goes to the without_gate folder
# and the with_gate images goes to the with_gate folder

import os
from pathlib import Path
import glob
import cv2 as cv
from resize import resize_and_pad

def prepare_folder(input_root: str, output_root: str):
    input_root_path = Path(input_root)
    output_root_path = Path(output_root)

    years = glob.glob("*", root_dir=input_root)

    for year in years:
        year_folder = input_root_path / year
        if not year_folder.is_dir():
            continue

        for subfolder in ['with_gate', 'without_gate']:
            input_folder = year_folder / subfolder
            output_folder = output_root_path / year / subfolder
            output_folder.mkdir(parents=True, exist_ok=True)

            for img_file in input_folder.glob('*.*'):
                if img_file.suffix.lower() != '.png':
                    continue  # skip non-image files

                # Read image
                image = cv.imread(str(img_file))
                if image is None:
                    print(f"Warning: Could not read image {img_file}, skipping.")
                    continue

                # Process image
                processed_image = resize_and_pad(image)

                # Save processed image
                output_path = output_folder / img_file.name
                cv.imwrite(str(output_path), processed_image)
                print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # D:\washik_personal\projects\gate_prediction\data\raw
    input_root = 'D:\\washik_personal\\projects\\gate_prediction\\data\\raw'
    output_root = 'D:\\washik_personal\\projects\\gate_prediction\\data\\processed'
    prepare_folder(input_root, output_root)
    