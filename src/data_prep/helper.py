import matplotlib.pyplot as plt
import torch

def plot(img_pts_list, point_size=40, point_color="red"):
    """
    img_pts_list: list of tuples (img, keypoints)
        img: PIL Image
        keypoints: KeyPoints object (N, K, 2)
    """

    num = len(img_pts_list)
    fig, axes = plt.subplots(1, num, figsize=(6 * num, 6))

    if num == 1:
        axes = [axes]

    for ax, (img, kpts) in zip(axes, img_pts_list):
        ax.imshow(img)
        ax.axis("off")

        # Convert supported keypoint containers into a tensor shaped (K, 2)
        if hasattr(kpts, "as_subclass"):
            pts = kpts.as_subclass(torch.Tensor)[0]  # KeyPoints: (N, K, 2) -> take first set
        else:
            pts_tensor = torch.as_tensor(kpts, dtype=torch.float32)
            if pts_tensor.ndim == 3:
                pts = pts_tensor[0]
            else:
                pts = pts_tensor

        xs = pts[:, 0].tolist()
        ys = pts[:, 1].tolist()

        ax.scatter(xs, ys, s=point_size, c=point_color)

    plt.show()
