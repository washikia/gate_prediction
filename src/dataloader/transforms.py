import os
from pathlib import Path
import glob
import cv2 as cv
from tools.helpers import get_gate_loc, add_label, save_tv_image
import numpy as np
from PIL import Image

from torchvision.transforms import v2
from torchvision import tv_tensors
import torch



class DeterministicZoom(v2.Transform):
    """
    Deterministic Zoom In / Zoom Out for tv_tensors.Image & tv_tensors.KeyPoints.

    zoom_factor > 1 → zoom in
    zoom_factor < 1 → zoom out
    """

    def __init__(self, zoom_factor: float):
        super().__init__()
        self.zoom_factor = zoom_factor

    def transform(self, inpt, params=None):
        if isinstance(inpt, tv_tensors.Image):
            return self._zoom_image(inpt)
        elif isinstance(inpt, tv_tensors.KeyPoints):
            return self._zoom_keypoints(inpt)
        return inpt

    def _zoom_image(self, img: tv_tensors.Image):
        # Call TV's deterministic affine transform
        return v2.functional.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=self.zoom_factor,
            shear=[0.0, 0.0],
            fill=0,      # black padding for zoom-out
            center=None,
        )

    def _zoom_keypoints(self, kp: tv_tensors.KeyPoints):
        return v2.functional.affine(
            kp,
            angle=0.0,
            translate=[0, 0],
            scale=self.zoom_factor,
            shear=[0.0, 0.0],
            fill=None,         # keypoints don't need fill
            center=None,
        )





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
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if not contours: 
            y, x, h, w = 0, 0, img_np.shape[0], img_np.shape[1]
        else:
            all_points = np.concatenate(contours)
            x, y, w, h = cv.boundingRect(all_points)
        
        cropped = img_np[y:y+h, x:x+w]

        # STORE crop for k-+-+eypoint transform
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
        resized = cv.resize(cropped, (target_w, target_h), interpolation=cv.INTER_CUBIC)

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

        # Verify size
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
        v2.Pad([15,29], fill=0),
        v2.RandomRotation((angle, angle)),
        aspectratio_preserving_Resize()
    ])

    # After transforms, labels may no longer be a KeyPoints subclass.
    # Convert to a plain numpy array, then to a nested Python list for JSON.
    img_lab_tuple = transforms(img_lab_tuple)

    transformed_img = img_lab_tuple["img"]
    transformed_labels = img_lab_tuple["labels"]
    
    return transformed_img, transformed_labels



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
            img_name = img_name.name    # keep this var: img_name -> it contains the extension .png

            # check if the image has a label
            label_path = input_root_path.parent / "labels" / "toy.json"
            label = get_gate_loc(label_path, img_name)
            assert label is not None


            # 1. save the grayscale image with same name without year
            img = mold_background_to_black(img)  # keep this var: img -> get other transforms from this
            save_path = output_path / img_name
            cv.imwrite(str(save_path), img)

            labels_kp = tv_tensors.KeyPoints(data=label, canvas_size=(256,512))
            img_name_ = img_name.replace('.png', '')  # remove extension


            # 2. fixed roation +8 of the image and labels
            img_rotated_p8, labels_rotated_p8 = transform_fixed_rotation(tv_tensors.Image(img), labels_kp, 8)
            save_name = img_name_ + "_rotated_p8.png"
            save_path_ = output_path / save_name
            save_tv_image(img_rotated_p8, save_path_)
            labels_list = np.asarray(labels_rotated_p8).tolist()
            add_label(label_path, save_name, labels_list)


            # 3. fixed roation -8 of the image and labels
            img_rotated_n8, labels_rotated_n8 = transform_fixed_rotation(tv_tensors.Image(img), labels_kp, -8)
            save_name = img_name_ + "_rotated_n8.png"
            save_path_ = output_path / save_name
            save_tv_image(img_rotated_n8, save_path_)
            labels_list = np.asarray(labels_rotated_n8).tolist()
            add_label(label_path, save_name, labels_list)


            # 4. fixed rotation +17 of the image and labels
            img_rotated_p17, labels_rotated_p17 = transform_fixed_rotation(tv_tensors.Image(img), labels_kp, 17)
            save_name = img_name_ + "_rotated_p17.png"
            save_path_ = output_path / save_name
            save_tv_image(img_rotated_p17, save_path_)
            labels_list = np.asarray(labels_rotated_p17).tolist()
            add_label(label_path, save_name, labels_list)


            # 5. fixed rotation -17 of the image and labels
            img_rotated_n17, labels_rotated_n17 = transform_fixed_rotation(tv_tensors.Image(img), labels_kp, -17)
            save_name = img_name_ + "_rotated_n17.png"
            save_path_ = output_path / save_name
            save_tv_image(img_rotated_n17, save_path_)
            labels_list = np.asarray(labels_rotated_n17).tolist()
            add_label(label_path, save_name, labels_list)


            # 6. fixed zoom-in 5% on original grayscale
            zoomin5 = DeterministicZoom(1.05)
            zoomedin5 = zoomin5({"img": tv_tensors.Image(img), "labels":labels_kp})
            save_name = img_name_ + "_zoomedin_5p.png"
            save_path = output_path / save_name
            save_tv_image(zoomedin5["img"], save_path)
            labels_list = np.asarray(zoomedin5["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 7. fixed zoom out 10% on original grayscale
            zoomout10 = DeterministicZoom(0.9)
            zoomedout10 = zoomout10({"img": tv_tensors.Image(img), "labels":labels_kp})
            save_name = img_name_ + "_zoomedout_10p.png"
            save_path = output_path / save_name
            save_tv_image(zoomedout10["img"], save_path)
            labels_list = np.asarray(zoomedout10["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 8. fixed zoom in 5% on (2) 8 degree anticlockwise rotation
            img_np = np.asarray(img_rotated_p8, dtype=np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]
            labels_list = np.asarray(labels_rotated_p8).tolist()
            
            zoomin5rotatedp8 = DeterministicZoom(1.05)
            zoomedin5rotatedp8 = zoomin5rotatedp8({"img": img_np, "labels": labels_list})
            save_name = img_name_ + "_zoomedinrotatedp8_5p.png"
            save_path = output_path / save_name
            save_tv_image(zoomedin5rotatedp8["img"], save_path)
            labels_list = np.asarray(zoomedin5rotatedp8["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 9. fixed zoom out 10% on (2) 8 degree anticlockwise rotation
            img_np = np.asarray(img_rotated_p8, dtype=np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]
            labels_list = np.asarray(labels_rotated_p8).tolist()
            
            zoomout10rotatedp8 = DeterministicZoom(0.9)
            zoomedout10rotatedp8 = zoomout10rotatedp8({"img": img_np, "labels": labels_list})
            save_name = img_name_ + "_zoomedoutrotatedp8_10p.png"
            save_path = output_path / save_name
            save_tv_image(zoomedout10rotatedp8["img"], save_path)
            labels_list = np.asarray(zoomedout10rotatedp8["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 10. fixed zoom in 5% on (3) 8 degree clockwise rotation
            img_np = np.asarray(img_rotated_n8, dtype=np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]
            labels_list = np.asarray(labels_rotated_n8).tolist()
            
            zoomin5rotatedn8 = DeterministicZoom(1.05)
            zoomedin5rotatedn8 = zoomin5rotatedn8({"img": img_np, "labels": labels_list})
            save_name = img_name_ + "_zoomedinrotatedn8_5p.png"
            save_path = output_path / save_name
            save_tv_image(zoomedin5rotatedn8["img"], save_path)
            labels_list = np.asarray(zoomedin5rotatedn8["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 11. fixed zoom out 10% on (3) 8 degree clockwise rotation
            img_np = np.asarray(img_rotated_n8, dtype=np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]
            labels_list = np.asarray(labels_rotated_n8).tolist()
            
            zoomout10rotatedn8 = DeterministicZoom(0.9)
            zoomedout10rotatedn8 = zoomout10rotatedn8({"img": img_np, "labels": labels_list})
            save_name = img_name_ + "_zoomedoutrotatedn8_10p.png"
            save_path = output_path / save_name
            save_tv_image(zoomedout10rotatedn8["img"], save_path)
            labels_list = np.asarray(zoomedout10rotatedn8["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 12. fixed zoom in 5% on (4) 17 degree anticlockwise rotation
            img_np = np.asarray(img_rotated_p17, dtype=np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]
            labels_list = np.asarray(labels_rotated_p17).tolist()
            
            zoom_in_5_rotated_p17 = DeterministicZoom(1.05)
            zoomed_in_5_rotated_p17 = zoom_in_5_rotated_p17({"img": img_np, "labels": labels_list})
            save_name = img_name_ + "_zoomed_in_rotated_p17_5p.png"
            save_path = output_path / save_name
            save_tv_image(zoomed_in_5_rotated_p17["img"], save_path)
            labels_list = np.asarray(zoomed_in_5_rotated_p17["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 13. fixed zoom out 10% on (4) 17 degree anticlockwise rotation
            img_np = np.asarray(img_rotated_p17, dtype=np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]
            labels_list = np.asarray(labels_rotated_p17).tolist()
            
            zoom_out_10_rotated_p17 = DeterministicZoom(0.9)
            zoomed_out_10_rotated_p17 = zoom_out_10_rotated_p17({"img": img_np, "labels": labels_list})
            save_name = img_name_ + "_zoomed_out_rotated_p17_10p.png"
            save_path = output_path / save_name
            save_tv_image(zoomed_out_10_rotated_p17["img"], save_path)
            labels_list = np.asarray(zoomed_out_10_rotated_p17["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 14. fixed zoom in 5% on (5) 17 degree clockwise rotation
            img_np = np.asarray(img_rotated_n17, dtype=np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]
            labels_list = np.asarray(labels_rotated_n17).tolist()
            
            zoom_in_5_rotated_n17 = DeterministicZoom(1.05)
            zoomed_in_5_rotated_n17 = zoom_in_5_rotated_n17({"img": img_np, "labels": labels_list})
            save_name = img_name_ + "_zoomed_in_rotated_n17_5p.png"
            save_path = output_path / save_name
            save_tv_image(zoomed_in_5_rotated_n17["img"], save_path)
            labels_list = np.asarray(zoomed_in_5_rotated_n17["labels"]).tolist()
            add_label(label_path, save_name, labels_list)


            # 15. fixed zoom out 10% on (5) 17 degree clockwise rotation
            img_np = np.asarray(img_rotated_n17, dtype=np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]
            labels_list = np.asarray(labels_rotated_n17).tolist()
            
            zoom_out_10_rotated_n17 = DeterministicZoom(0.9)
            zoomed_out_10_rotated_n17 = zoom_out_10_rotated_n17({"img": img_np, "labels": labels_list})
            save_name = img_name_ + "_zoomed_out_rotated_n17_10p.png"
            save_path = output_path / save_name
            save_tv_image(zoomed_out_10_rotated_n17["img"], save_path)
            labels_list = np.asarray(zoomed_out_10_rotated_n17["labels"]).tolist()
            add_label(label_path, save_name, labels_list)




            
        print("passed")




if __name__ == "__main__":
    generate_transformed_dataset("D:\\washik_personal\\projects\\gate_prediction\\data\\toy")

