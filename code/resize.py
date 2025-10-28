import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

im = cv.imread("D:\\washik_personal\\projects\\gate_prediction\\data\\raw\\2025\\with_gate\\25M1750D_front.png")
assert im is not None


#  function: resize_and_pad
def resize_and_pad(image, target_size=(512, 512), pad_color=(255, 255, 255)):
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

    top_bottom_pad = (target_size[1] - target_height) // 2
    extra_bottom = target_size[1] - (target_height + (top_bottom_pad * 2))  # Handle odd numbers

    # Add padding
    padded = cv.copyMakeBorder(
        resized,
        top_bottom_pad,
        top_bottom_pad + extra_bottom,
        left_right_pad,
        left_right_pad,
        cv.BORDER_CONSTANT,
        value=pad_color
    )

    # Verify size
    assert padded.shape[:2] == target_size, \
        f"Expected size {target_size}, got {padded.shape[:2]}"
    
    return padded


# Keep an RGB copy for the "original" display
original_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
edges = cv.Canny(imgray, 50, 200)
contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# handle case with no contours
if not contours:
    h, w = im.shape[:2]
    x, y, w, h = 0, 0, w, h
else:
    all_points = np.concatenate(contours)
    x, y, w, h = cv.boundingRect(all_points)

# Draw the bounding box on the original image copy
boxed = im.copy()
cv.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 1)
boxed_rgb = cv.cvtColor(boxed, cv.COLOR_BGR2RGB)

# Crop the region
cropped_bgr = im[y:y + h, x:x + w]
cropped_rgb = cv.cvtColor(cropped_bgr, cv.COLOR_BGR2RGB)

# Step 5: Resize preserving aspect ratio with width=470
target_width = 470
aspect_ratio = cropped_bgr.shape[1] / cropped_bgr.shape[0]
target_height = int(target_width / aspect_ratio)
resized = cv.resize(cropped_bgr, (target_width, target_height))

# Step 6: Add padding to make final size 512x512
# Calculate padding
left_right_pad = 21  # 21px each side
total_width = target_width + (left_right_pad * 2)  # Should be 512
needed_height = 512
top_bottom_pad = (needed_height - target_height) // 2
extra_bottom = needed_height - (target_height + (top_bottom_pad * 2))  # Handle odd numbers

# Add padding
padded = cv.copyMakeBorder(
    resized,
    top_bottom_pad,  # top
    top_bottom_pad + extra_bottom,  # bottom
    left_right_pad,  # left
    left_right_pad,  # right
    cv.BORDER_CONSTANT,
    value=[255, 255, 255]  # white padding instead of black
)

# Convert for display
padded_rgb = cv.cvtColor(padded, cv.COLOR_BGR2RGB)

# Display original, boxed, and cropped side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(original_rgb)
axes[0].set_title("Original")
axes[0].axis("off")
print("image shape: ", original_rgb.shape)

axes[1].imshow(boxed_rgb)
axes[1].set_title("With Bounding Box")
axes[1].axis("off")
print("image shape: ", boxed_rgb.shape)

axes[2].imshow(cropped_rgb)
axes[2].set_title("Cropped")
axes[2].axis("off")
print("image shape: ", cropped_rgb.shape)

plt.tight_layout()
plt.show()

# Update visualization to show all steps
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
axes[0].imshow(original_rgb)
axes[0].set_title(f"Original\n{original_rgb.shape}")
axes[0].axis("off")

axes[1].imshow(cropped_rgb)
axes[1].set_title(f"Cropped\n{cropped_rgb.shape}")
axes[1].axis("off")

resized_rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
axes[2].imshow(resized_rgb)
axes[2].set_title(f"Resized (w=470)\n{resized_rgb.shape}")
axes[2].axis("off")

axes[3].imshow(padded_rgb)
axes[3].set_title(f"Padded (512x512)\n{padded_rgb.shape}")
axes[3].axis("off")

plt.tight_layout()
plt.show()

# Verify final dimensions
assert padded.shape == (512, 512, 3), f"Expected (512, 512, 3), got {padded.shape}"