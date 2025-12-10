from cv2.typing import MatLike


from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import torch
from tools.helpers import get_gate_loc, add_label
from torchvision import tv_tensors
from torchvision.transforms import v2
# from helper import plot
import json






class aspectratio_preserving_Resize(v2.Transform):
    '''
    TorchVision v2-compatible transform that can be passed to v2.Compose
    It Resizes Image and KeyPoints only, and preserves aspect ratio
    '''

    def __init__(self):
        super().__init__()
        self.final_h, self.final_w = 256, 512
        self.target_height, self.target_width = None, None
        self.crop_x, self.crop_y, self.crop_w, self.crop_h, self.left_pad, self.top_pad = None, None, None, None, None, None

    # "In order to support arbitrary inputs ...  override the .transform() method"
    def transform(self, inpt, params=None):
        # Handle dictionaries (from v2.Compose)
        # if isinstance(inpt, dict):
        #     result = inpt.copy()  # Preserve other keys
        #     # Transform image first to store crop/pad state
        #     if "img" in inpt:
        #         result["img"] = self._transform_image(inpt["img"])
        #     # Then transform labels using the stored state
        #     if "labels" in inpt:
        #         result["labels"] = self._transfrom_keypoints(inpt["labels"])
        #     return result
        if isinstance(inpt, tv_tensors.Image):
            return self._transform_image(inpt)
        elif isinstance(inpt, tv_tensors.KeyPoints):
            return self._transform_keypoints(inpt)
        else:
            return inpt


    def _transform_image(self, img: tv_tensors.Image):
        img_np = img.numpy()

        if img_np.ndim == 3 and img_np.shape[0] == 1:
            img_np = img_np[0]  # Now shape = (H, W)

        edges = cv.Canny(img_np, 50, 250)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if not contours: 
            y, x, h, w = 0, 0, img_np.shape[0], img_np.shape[1]
        else:
            all_points = np.concatenate(contours)
            x, y, w, h = cv.boundingRect(all_points)
        
        cropped = img_np[y:y+h, x:x+w]

        # STORE crop for keypoint transform
        self.crop_x, self.crop_y = x, y
        self.crop_w, self.crop_h = w, h

        # Resize logic
        needed_h, needed_w = self.final_h, self.final_w

        if h <= needed_h:
            target_w = 470
            aspect_ratio = w / max(1, h)
            target_h = max(1, int(round(target_w/aspect_ratio)))

            if target_h > needed_h:
                scale = needed_h / float(target_h)
                target_h = needed_h
                target_w = max(1, int(round(target_w * scale)))
            
        else:
            target_h = 240
            aspect_ratio = w / max(1, h)
            target_w = max(1, int(round(target_h * aspect_ratio)))

            if target_w > needed_w:
                scale = needed_w / float(target_w)
                target_w = needed_w
                target_h = max(1, int(round(target_h * scale)))
        
        self.target_height, self.target_width = target_h, target_w
        resized = cv.resize(cropped, (target_w, target_h), interpolation=cv.INTER_NEAREST)

        left_right_total = max(0, needed_w - target_w)
        left_right_pad = left_right_total // 2
        extra_right = left_right_total - (left_right_pad * 2)

        top_bottom_total = max(0, needed_h -target_h)
        top_bottom_pad = top_bottom_total // 2
        extra_bottom = top_bottom_total - (top_bottom_pad * 2)

        # STORE padding offsets
        self.left_pad = left_right_pad
        self.top_pad = top_bottom_pad

        
        padded = cv.copyMakeBorder(
            resized,
            top_bottom_pad,
            top_bottom_pad + extra_bottom,
            left_right_pad,
            left_right_pad + extra_right,
            cv.BORDER_CONSTANT,
            value=(0,0,0)
        )

        padded_tensor = torch.from_numpy(padded).unsqueeze(0)  # C, H, W

        return tv_tensors.Image(padded_tensor)

    
    def _transform_keypoints(self, keypoints: tv_tensors.KeyPoints):
        """
        Keypoints are in the format: [[x, y], [x2, y2], ...]
        Must apply same crop → resize → pad pipeline as image.
        """

        # Convert to numpy for easier math
        kp = np.asarray(keypoints, dtype=np.float32)

        # --- 1) Subtract crop origin ---
        kp[:, 0] -= self.crop_x
        kp[:, 1] -= self.crop_y

        # --- 2) Resize scaling ---
        scale_x = self.target_width  / max(1, self.crop_w)
        scale_y = self.target_height / max(1, self.crop_h)

        kp[:, 0] *= scale_x
        kp[:, 1] *= scale_y

        # --- 3) Add padding offsets ---
        kp[:, 0] += self.left_pad
        kp[:, 1] += self.top_pad

        # Return as KeyPoints again
        return tv_tensors.KeyPoints(kp, canvas_size=(256, 512))



def transform_fixed_rotation(image: tv_tensors.Image, labels: tv_tensors.KeyPoints, angle: int):
    '''
    Pads, rotates, and resizes image and labels to prevent mold from being cut.
    
    Args:
        image: Input grayscale image as tv_tensors.Image
        labels: KeyPoints tensor with gate coordinates
    
    Returns:
        tuple: (transformed_image: tv_tensors.Image, transformed_labels: tv_tensors.KeyPoints)
    '''
    img_lab_tuple = {
        "img": image,
        "labels": labels
    }

    transforms = v2.Compose([
        v2.Pad([15,29], fill=0),
        v2.RandomRotation((angle, angle)),
        aspectratio_preserving_Resize()
    ])

    img_lab_tuple = transforms(img_lab_tuple)

    # Ensure the results are tv_tensors.Image and tv_tensors.KeyPoints
    transformed_img = img_lab_tuple["img"]
    transformed_labels = img_lab_tuple["labels"]
    
    # # Convert to tv_tensors if they're not already (some transforms may return plain tensors)
    # if not isinstance(transformed_img, tv_tensors.Image):
    #     transformed_img = tv_tensors.Image(transformed_img)
    
    # if not isinstance(transformed_labels, tv_tensors.KeyPoints):
    #     # Convert to KeyPoints with the correct canvas size
    #     labels_arr = np.asarray(transformed_labels)
    #     transformed_labels = tv_tensors.KeyPoints(labels_arr, canvas_size=(256, 512))

    return transformed_img, transformed_labels



if __name__ == "__main__":
    data_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\labels\\annotations.json'
    image_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\processed\\2025\\without_gate\\21M4870D_front.png'
    # gate_loc = get_gate_loc(data_path, image_path)
    # print(gate_loc)
    # plot([(Image.open(image_path), gate_loc)])
    ip = 'D:\\washik_personal\\projects\\gate_prediction\\data\\toy\\2021\\without_gate\\21M4870D_front.png'
    label_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\labels\\toy.json'
    pil_img = cv.imread(str(ip), cv.IMREAD_GRAYSCALE)
    print("before tv_tensors.Image: ", pil_img.shape)
    # pil_img = np.squeeze(pil_img)  # Remove leading dimension if present
    labels = get_gate_loc(label_path, "21M4870D_front.png")
    labels = tv_tensors.KeyPoints(data=labels, canvas_size=(256,512))
    img_lab_tuple = {
        "img": tv_tensors.Image(pil_img),
        "labels": labels
    }
    print(img_lab_tuple["img"].shape, img_lab_tuple["labels"])
    transform = aspectratio_preserving_Resize()
    transformed = transform(img_lab_tuple)
    print(type(transformed["img"]), transformed["img"].shape)
    print(type(transformed["labels"]), transformed["labels"])

    print("\n\n transform_fixed_rotation \n\n")
    img, label = transform_fixed_rotation(tv_tensors.Image(pil_img), labels, 10)

    print(type(img), img.shape)
    print(type(label), label)
