import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision import tv_tensors
from tools.helpers import get_gate_loc
from transforms import transforms


def make_dataset(data_path: str, label_path: str):
    '''
    make a pandas dataframe the contains the image path and the labels
    the strucutre of the image path is like this:
    processed
    |-- 2025
    |   |-- without_gate
    |   |   |-- 25M1710D_front.png
    |   |   |-- 25M1710D_front.png
    |   |-- with_gate
    |   |   |-- 25M1710D_front.png
    |   |   |-- 25M1710D_front.png
    ...
    the label is a json file that contains the gate location for each image
    '''
    rows = []

    for year in os.listdir(data_path):
        year_path = os.path.join(data_path, year, "without_gate")
        if not os.path.isdir(year_path):
            continue

        for image in os.listdir(year_path):
            image_path = os.path.join(year_path, image)
            assert os.path.isfile(image_path)
            label = get_gate_loc(label_path, image)

            rows.append({
                "image_path": image_path,
                "label": label
            })

    df = pd.DataFrame(rows)
    return df



class GateDataset(Dataset):
    def __init__(self, image_root_path, label_path, transform=None) -> None:
        super().__init__()
        self.image_root_path = image_root_path
        self.label_path = label_path
        self.df = make_dataset(self.image_root_path, self.label_path)
        self.image_paths = self.df.iloc[:, 0]
        self.labels = self.df.iloc[:, 1]
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = decode_image(self.image_paths[idx])     # it returns (C, H, W) -> this might be an issue later
        label = self.labels[idx]
        pair = {
            "img": tv_tensors.Image(image), 
            "label": tv_tensors.KeyPoints(label, canvas_size=(256, 512))}
        if self.transform:
            pair = self.transform(pair)
        
        return pair["img"], pair["label"]





if __name__ == "__main__":
    data_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\toy'
    label_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\labels\\annotations.json'
    df = make_dataset(data_path, label_path)
    print(df.head())
    print(df.tail())
    print(df.shape)