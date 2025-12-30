import pytest
import torch
from torchvision import tv_tensors
from torch.utils.data import DataLoader

from src.dataloader.dataset import GateDataset
from src.dataloader.transforms import transforms


DATA_PATH = 'D:\\washik_personal\\projects\\gate_prediction\\data\\toy'
LABEL_PATH = 'D:\\washik_personal\\projects\\gate_prediction\\data\\labels\\annotations.json'


@pytest.fixture
def dataset_no_transform():
    return GateDataset(
        image_root_path=DATA_PATH,
        label_path=LABEL_PATH,
        transform=None
    )


@pytest.fixture
def dataset_with_transform():
    return GateDataset(
        image_root_path=DATA_PATH,
        label_path=LABEL_PATH,
        transform=transforms
    )


# ------------------------
# Dataset basics
# ------------------------

def test_dataset_length(dataset_no_transform):
    assert len(dataset_no_transform) > 0


def test_first_sample_types(dataset_no_transform):
    img, label = dataset_no_transform[0]

    assert isinstance(img, tv_tensors.Image)
    assert isinstance(label, tv_tensors.KeyPoints)


def test_image_shape(dataset_no_transform):
    img, _ = dataset_no_transform[0]

    assert img.ndim == 3  # C,H,W
    assert img.shape[0] in (1, 3)  # grayscale or RGB


def test_label_shape(dataset_no_transform):
    _, label = dataset_no_transform[0]

    assert label.ndim == 2
    assert label.shape[1] == 2  # (x, y)
    assert label.shape[0] > 0   # at least one gate


# ------------------------
# Bounds safety
# ------------------------

def test_keypoints_within_image(dataset_no_transform):
    img, label = dataset_no_transform[0]
    _, H, W = img.shape

    xs = label[:, 0]
    ys = label[:, 1]

    assert torch.all(xs >= 0)
    assert torch.all(xs < W)
    assert torch.all(ys >= 0)
    assert torch.all(ys < H)


# ------------------------
# Transform tests
# ------------------------

def test_transform_runs(dataset_with_transform):
    img, label = dataset_with_transform[0]

    assert isinstance(img, tv_tensors.Image)
    assert isinstance(label, tv_tensors.KeyPoints)


def test_transform_output_size(dataset_with_transform):
    img, _ = dataset_with_transform[0]

    # Expected model input size
    assert img.shape[1:] == (256, 512)


# ------------------------
# DataLoader test
# ------------------------

def test_dataloader_batching(dataset_with_transform):
    loader = DataLoader(
        dataset_with_transform,
        batch_size=4,
        shuffle=False
    )

    imgs, labels = next(iter(loader))

    assert imgs.shape[0] == 4
    assert labels.shape[0] == 4
