import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



#  function: resize_and_pad
def resize_and_pad(image):
    """
    Resize and pad an image following the gate detection pipeline.
    
    Args:
        image: BGR input image
        target_size: tuple of (width, height) for final size, default (512, 512)
        pad_color: tuple of (B, G, R) for padding color, default white
    
    Returns:
        padded: BGR image of size target_size with white padding
    """
    # Input validation
    if image is None or image.size == 0:
        raise ValueError("Empty or invalid input image")

    # Steps 1-4: Find bounding box
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(imgray, 50, 250)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = image.shape[:2]
        x, y, w, h = 0, 0, w, h
    else:
        all_points = np.concatenate(contours)
        x, y, w, h = cv.boundingRect(all_points)

    # Crop to bounding box
    cropped_bgr = image[y:y + h, x:x + w]

    # Step 5: Resize to width 470 (not target_size[0] - padding)
    target_width = 470  # Fixed width as per requirements
    aspect_ratio = cropped_bgr.shape[1] / cropped_bgr.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized = cv.resize(cropped_bgr, (target_width, target_height))

    # Step 6: Add padding to reach 512x512
    left_right_pad = 21  # Fixed padding as per requirements
    needed_height = 256
    top_bottom_pad = (needed_height - target_height) // 2
    extra_bottom = needed_height - (target_height + (top_bottom_pad * 2))

    padded = cv.copyMakeBorder(
        resized,
        top_bottom_pad,
        top_bottom_pad + extra_bottom,
        left_right_pad,
        left_right_pad,
        cv.BORDER_CONSTANT,
        value=(255,255,255)
    )

    # Verify size
    assert padded.shape[:2] == (256, 512), \
        f"Expected size {(256, 512)}, got {padded.shape[:2]}"
    
    return padded



# -----------------------
# Tests and visualizations
# -----------------------

def _compute_bbox_steps(image):
    """Duplicate detection steps externally for visualization (original, boxed, cropped, resized, padded)."""
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(imgray, 50, 250)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        x, y, w, h = 0, 0, w, h
    else:
        all_points = np.concatenate(contours)
        x, y, w, h = cv.boundingRect(all_points)
    boxed = image.copy()
    cv.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cropped = image[y:y + h, x:x + w]
    if cropped.size == 0:
        cropped = image.copy()
    # resized (same logic as function)
    target_width = 470
    aspect_ratio = cropped.shape[1] / max(1, cropped.shape[0])
    target_height = max(1, int(target_width / aspect_ratio))
    resized = cv.resize(cropped, (target_width, target_height), interpolation=cv.INTER_LINEAR)
    padded = resize_and_pad(image)  # final output
    return {"original": image, "boxed": boxed, "cropped": cropped, "resized": resized, "padded": padded}

def visual_test(img_path):
    if img_path:
        im = cv.imread(img_path)
        print("Using image:", img_path)
    else:
        print("No example image found, using synthetic.")
        im = np.full((600, 800, 3), 200, dtype=np.uint8)
        cv.rectangle(im, (150, 120), (650, 480), (0, 0, 0), -1)

    steps = _compute_bbox_steps(im)

    def to_rgb(bgr): return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(to_rgb(steps["original"]))
    axes[0].set_title(f"Original\n{steps['original'].shape}")
    axes[0].axis("off")

    axes[1].imshow(to_rgb(steps["boxed"]))
    axes[1].set_title(f"With BBox\n{steps['boxed'].shape}")
    axes[1].axis("off")

    axes[2].imshow(to_rgb(steps["cropped"]))
    axes[2].set_title(f"Cropped\n{steps['cropped'].shape}")
    axes[2].axis("off")

    axes[3].imshow(to_rgb(steps["resized"]))
    axes[3].set_title(f"Resized (470w)\n{steps['resized'].shape}")
    axes[3].axis("off")

    axes[4].imshow(to_rgb(steps["padded"]))
    axes[4].set_title(f"Padded (512x512)\n{steps['padded'].shape}")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()

def test_synthetic():
    # basic functional test using synthetic image
    img = np.full((600, 800, 3), 200, dtype=np.uint8)
    cv.rectangle(img, (150, 120), (650, 480), (0, 0, 0), -1)
    out = resize_and_pad(img)
    assert out.shape == (512, 512, 3)
    # check left padding is white
    assert np.all(out[:, 0] == 255)
    print("test_synthetic: PASS")

if __name__ == "__main__":
    image_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\raw\\2025\\without_gate\\25M1710D_front.png'
    test_synthetic()
    visual_test(image_path)