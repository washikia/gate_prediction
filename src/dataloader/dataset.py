import pandas as pd
import os

from tools.helpers import get_gate_loc


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
            label = get_gate_loc(label_path, image)

            rows.append({
                "image_path": image_path,
                "label": label
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    data_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\processed'
    label_path = 'D:\\washik_personal\\projects\\gate_prediction\\data\\labels\\annotations.json'
    df = make_dataset(data_path, label_path)
    print(df.head())
    print(df.tail())
    print(df.shape)