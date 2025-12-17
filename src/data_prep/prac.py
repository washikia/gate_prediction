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






# class aspectratio_preserving_Resize(v2.Transform):
#     '''
#     TorchVision v2-compatible transform that can be passed to v2.Compose
#     It Resizes Image and KeyPoints only, and preserves aspect ratio
#     '''

#     def __init__(self):
#         super().__init__()
#         self.final_h, self.final_w = 256, 512
#         self.target_height, self.target_width = None, None
#         self.crop_x, self.crop_y, self.crop_w, self.crop_h, self.left_pad, self.top_pad = None, None, None, None, None, None

#     # "In order to support arbitrary inputs ...  override the .transform() method"
#     def transform(self, inpt, params=None):
#         # Handle dictionaries (from v2.Compose)
#         # if isinstance(inpt, dict):
#         #     result = inpt.copy()  # Preserve other keys
#         #     # Transform image first to store crop/pad state
#         #     if "img" in inpt:
#         #         result["img"] = self._transform_image(inpt["img"])
#         #     # Then transform labels using the stored state
#         #     if "labels" in inpt:
#         #         result["labels"] = self._transfrom_keypoints(inpt["labels"])
#         #     return result
#         if isinstance(inpt, tv_tensors.Image):
#             return self._transform_image(inpt)
#         elif isinstance(inpt, tv_tensors.KeyPoints):
#             return self._transform_keypoints(inpt)
#         else:
#             return inpt


#     def _transform_image(self, img: tv_tensors.Image):
#         img_np = img.numpy()

#         if img_np.ndim == 3 and img_np.shape[0] == 1:
#             img_np = img_np[0]  # Now shape = (H, W)

#         edges = cv.Canny(img_np, 50, 250)
#         contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#         if not contours: 
#             y, x, h, w = 0, 0, img_np.shape[0], img_np.shape[1]
#         else:
#             all_points = np.concatenate(contours)
#             x, y, w, h = cv.boundingRect(all_points)
        
#         cropped = img_np[y:y+h, x:x+w]

#         # STORE crop for keypoint transform
#         self.crop_x, self.crop_y = x, y
#         self.crop_w, self.crop_h = w, h

#         # Resize logic
#         needed_h, needed_w = self.final_h, self.final_w

#         if h <= needed_h:
#             target_w = 470
#             aspect_ratio = w / max(1, h)
#             target_h = max(1, int(round(target_w/aspect_ratio)))

#             if target_h > needed_h:
#                 scale = needed_h / float(target_h)
#                 target_h = needed_h
#                 target_w = max(1, int(round(target_w * scale)))
            
#         else:
#             target_h = 240
#             aspect_ratio = w / max(1, h)
#             target_w = max(1, int(round(target_h * aspect_ratio)))

#             if target_w > needed_w:
#                 scale = needed_w / float(target_w)
#                 target_w = needed_w
#                 target_h = max(1, int(round(target_h * scale)))
        
#         self.target_height, self.target_width = target_h, target_w
#         resized = cv.resize(cropped, (target_w, target_h), interpolation=cv.INTER_NEAREST)

#         left_right_total = max(0, needed_w - target_w)
#         left_right_pad = left_right_total // 2
#         extra_right = left_right_total - (left_right_pad * 2)

#         top_bottom_total = max(0, needed_h -target_h)
#         top_bottom_pad = top_bottom_total // 2
#         extra_bottom = top_bottom_total - (top_bottom_pad * 2)

#         # STORE padding offsets
#         self.left_pad = left_right_pad
#         self.top_pad = top_bottom_pad

        
#         padded = cv.copyMakeBorder(
#             resized,
#             top_bottom_pad,
#             top_bottom_pad + extra_bottom,
#             left_right_pad,
#             left_right_pad + extra_right,
#             cv.BORDER_CONSTANT,
#             value=(0,0,0)
#         )

#         padded_tensor = torch.from_numpy(padded).unsqueeze(0)  # C, H, W

#         return tv_tensors.Image(padded_tensor)

    
#     def _transform_keypoints(self, keypoints: tv_tensors.KeyPoints):
#         """
#         Keypoints are in the format: [[x, y], [x2, y2], ...]
#         Must apply same crop → resize → pad pipeline as image.
#         """

#         # Convert to numpy for easier math
#         kp = np.asarray(keypoints, dtype=np.float32)

#         # --- 1) Subtract crop origin ---
#         kp[:, 0] -= self.crop_x
#         kp[:, 1] -= self.crop_y

#         # --- 2) Resize scaling ---
#         scale_x = self.target_width  / max(1, self.crop_w)
#         scale_y = self.target_height / max(1, self.crop_h)

#         kp[:, 0] *= scale_x
#         kp[:, 1] *= scale_y

#         # --- 3) Add padding offsets ---
#         kp[:, 0] += self.left_pad
#         kp[:, 1] += self.top_pad

#         # Return as KeyPoints again
#         return tv_tensors.KeyPoints(kp, canvas_size=(256, 512))



# def transform_fixed_rotation(image: tv_tensors.Image, labels: tv_tensors.KeyPoints, angle: int):
#     '''
#     Pads, rotates, and resizes image and labels to prevent mold from being cut.
    
#     Args:
#         image: Input grayscale image as tv_tensors.Image
#         labels: KeyPoints tensor with gate coordinates
    
#     Returns:
#         tuple: (transformed_image: tv_tensors.Image, transformed_labels: tv_tensors.KeyPoints)
#     '''
#     img_lab_tuple = {
#         "img": image,
#         "labels": labels
#     }

#     transforms = v2.Compose([
#         v2.Pad([15,29], fill=0),
#         v2.RandomRotation((angle, angle)),
#         aspectratio_preserving_Resize()
#     ])

#     img_lab_tuple = transforms(img_lab_tuple)

#     # Ensure the results are tv_tensors.Image and tv_tensors.KeyPoints
#     transformed_img = img_lab_tuple["img"]
#     transformed_labels = img_lab_tuple["labels"]
    
#     # # Convert to tv_tensors if they're not already (some transforms may return plain tensors)
#     # if not isinstance(transformed_img, tv_tensors.Image):
#     #     transformed_img = tv_tensors.Image(transformed_img)
    
#     # if not isinstance(transformed_labels, tv_tensors.KeyPoints):
#     #     # Convert to KeyPoints with the correct canvas size
#     #     labels_arr = np.asarray(transformed_labels)
#     #     transformed_labels = tv_tensors.KeyPoints(labels_arr, canvas_size=(256, 512))

#     return transformed_img, transformed_labels


def flip_keypoints(kp, canvas_size, direction="horizontal"):
    """
    kp: numpy array of shape (N, 2) or list of [[x, y], ...]
    canvas_size: (H, W)
    direction: "horizontal" or "vertical"

    Returns the flipped keypoints as an ndarray (N, 2).
    """

    kp = np.asarray(kp, dtype=np.float32)
    H, W = canvas_size

    if direction == "horizontal":
        # x' = W - 1 - x
        flipped = kp.copy()
        flipped[:, 0] = (W - 1) - kp[:, 0]

    elif direction == "vertical":
        # y' = H - 1 - y
        flipped = kp.copy()
        flipped[:, 1] = (H - 1) - kp[:, 1]

    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    return flipped


class DeterministicFlip(v2.Transform):
    def __init__(self, direction="horizontal"):
        super().__init__()
        assert direction in ("horizontal", "vertical")
        self.direction = direction

    def transform(self, inpt, params):
        # Case 1: Image
        if isinstance(inpt, tv_tensors.Image) or isinstance(inpt, tv_tensors.KeyPoints):
            if self.direction == "horizontal":
                return v2.functional.horizontal_flip(inpt)
            else:
                return v2.functional.vertical_flip(inpt)

        return inpt


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



class RandomKeyPointJitter(v2.Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, inpt, params):
        if isinstance(inpt, tv_tensors.KeyPoints):
            inpt_list = np.asarray(inpt).tolist()
            transformed_keypoints = []
            for keypoint in inpt_list:
                assert len(keypoint) == 2
                new_keypoint = transform_point(keypoint)
                transformed_keypoints.append(new_keypoint)
            return tv_tensors.KeyPoints(transformed_keypoints, canvas_size=inpt.canvas_size)
        return inpt



if __name__ == "__main__":
    orli = [[1,2],[2,3],[3,4],[4,5],[5,6]]
    orli_kp = tv_tensors.KeyPoints(orli, canvas_size=(256,512))
    transform = RandomKeyPointJitter()
    output = transform(orli_kp)
    print(output)

    