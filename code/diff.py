import cv2
import numpy as np
import matplotlib.pyplot as plt


def align_images(img_ref, img_mov, max_features=500, good_match_percent=0.15):
    # Convert to grayscale if needed
    img1_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if img_ref.ndim==3 else img_ref
    img2_gray = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY) if img_mov.ndim==3 else img_mov

    # Display input images
    plt.figure(figsize=(10,4))
    plt.suptitle("Input Images", fontsize=16)
    plt.subplot(1,2,1)
    plt.title("Reference (gray)")
    plt.imshow(img1_gray, cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Moving (gray)")
    plt.imshow(img2_gray, cmap='gray')
    plt.axis('off')
    plt.show()

    # ORB feature detection
    orb = cv2.ORB_create(max_features)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)

    # Draw keypoints
    img1_kp = cv2.drawKeypoints(img1_gray, k1, None, color=(0,255,0))
    img2_kp = cv2.drawKeypoints(img2_gray, k2, None, color=(0,255,0))
    plt.figure(figsize=(10,4))
    plt.suptitle("Keypoints Detection", fontsize=16)
    plt.subplot(1,2,1)
    plt.title("Ref Keypoints")
    plt.imshow(img1_kp)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Mov Keypoints")
    plt.imshow(img2_kp)
    plt.axis('off')
    plt.show()

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)
    num_good = int(len(matches) * good_match_percent)
    matches = matches[:num_good]

    # Draw matches
    img_matches = cv2.drawMatches(img1_gray, k1, img2_gray, k2, matches, None, flags=2)
    plt.figure(figsize=(15,6))
    plt.suptitle(f"Top {num_good} Matches", fontsize=16)
    plt.title(f"Top {num_good} Matches")
    plt.imshow(img_matches)
    plt.axis('off')
    plt.show()

    # Extract matched points
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    if len(pts1) < 4:
        print("Not enough matches for homography.")
        return img_mov  # fallback: not enough matches

    # Homography
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    h, w = img_ref.shape[:2]
    aligned = cv2.warpPerspective(img_mov, H, (w, h))

    # Show aligned image
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Reference")
    plt.imshow(img_ref if img_ref.ndim==2 else cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Aligned Moving")
    plt.imshow(aligned if aligned.ndim==2 else cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return aligned


# def align_images(img_ref, img_mov, max_features=500, good_match_percent=0.15):
#     # ORB feature-based alignment (returns aligned moving image)
#     img1_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if img_ref.ndim==3 else img_ref
#     img2_gray = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY) if img_mov.ndim==3 else img_mov

#     orb = cv2.ORB_create(max_features)
#     k1, d1 = orb.detectAndCompute(img1_gray, None)
#     k2, d2 = orb.detectAndCompute(img2_gray, None)
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     matches = bf.match(d1, d2)
#     matches = sorted(matches, key=lambda x: x.distance)
#     num_good = int(len(matches) * good_match_percent)
#     matches = matches[:num_good]
#     pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
#     pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
#     if len(pts1) < 4:
#         return img_mov  # fallback: not enough matches
#     H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
#     h, w = img_ref.shape[:2]
#     aligned = cv2.warpPerspective(img_mov, H, (w, h))
#     return aligned


def detect_gate_coords(img_with, img_without, preprocess_blur=1, diff_thresh=30,
                       min_area=20, max_area=5000, morph_kernel=3, use_registration=True):
    # ensure grayscale single-channel numpy arrays
    if img_with.ndim == 3:
        w = cv2.cvtColor(img_with, cv2.COLOR_BGR2GRAY)
    else:
        w = img_with.copy()
    if img_without.ndim == 3:
        wo = cv2.cvtColor(img_without, cv2.COLOR_BGR2GRAY)
    else:
        wo = img_without.copy()

    # optional registration
    if use_registration:
        try:
            aligned = align_images(wo, w)  # align moving=w to ref=wo
            w = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim==3 else aligned
        except Exception as e:
            pass

    # normalize / blur
    w = cv2.GaussianBlur(w, (3,3), preprocess_blur)
    wo = cv2.GaussianBlur(wo, (3,3), preprocess_blur)

    # absolute difference
    diff = cv2.absdiff(w, wo)

    # threshold
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    # morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # optional refine with Canny of diff
    # edges = cv2.Canny(diff, 50, 150)
    # mask = cv2.bitwise_or(mask, edges)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gate_coords = []
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w_box//2, y + h_box//2
        gate_coords.append((cx, cy))
        bboxes.append((x, y, w_box, h_box))

    return gate_coords, mask, bboxes

# Usage example:
img_with = cv2.imread("25M1710D_front.png", cv2.IMREAD_GRAYSCALE)
img_without = cv2.imread("25M1710D_front_gates.png", cv2.IMREAD_GRAYSCALE)
coords, mask, boxes = detect_gate_coords(img_with, img_without, use_registration=True)
print("Detected gate coords:", coords)
cv2.imwrite("diff_mask.png", mask)
