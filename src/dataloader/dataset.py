import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision import tv_tensors
from tools.helpers import get_gate_loc, show_images_and_gates
from src.dataloader.transforms import transforms
from random import randint



def make_dataset_2(data_path: str, label_path: str):
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


def make_dataset(data_path: str, label_path: str):
    rows = []
    for image in os.listdir(data_path):
            image_path = os.path.join(data_path, image)
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
    data_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\final_dataset'
    label_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\labels\\toy.json'
    dataset = GateDataset(image_root_path=data_path, label_path=label_path, transform=transforms)
    print(len(dataset))
    idxs = [randint(0, len(dataset)-1) for _ in range(5)]
    
    labels_list = []
    # print(type(idxs[0]["img"]))
    print(type(dataset[idxs[1]][1]))
    images_list = []
    for x in idxs:
        images_list.append(dataset[x][0].permute(1,2,0).numpy())
        labels_list.append(dataset[x][1])


    # show_images(images_list)
    show_images_and_gates(images_list, labels_list)



