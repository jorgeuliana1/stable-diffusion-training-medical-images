import os
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, Features, Image as HFImage, Value

from .my_transforms import CropCenterSquare, RandomHorizontalFlip, RandomRotation  # Assuming these are defined in my_transforms
import multiprocessing as mp

class MyDataset(Dataset):
    def __init__(self, root: str, csv_train: str, csv_test: str, train: bool = True):
        self.trainsize = (224, 224)
        self.train = train
        self.root = root

        # Defining .csv file to be used
        csv_name = csv_train if train else csv_test

        # Opening dataframe:
        self.df = pd.read_csv(csv_name, header=0)

        # For generator functionality:
        self.curr_idx = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        raise NotImplementedError("Define the __getitem__ method for your class")

    def __next__(self):
        value = self[self.curr_idx]
        self.curr_idx += 1
        return value

    @property
    def labels_balance(self):
        y_series = self.df[self.y]
        v_counts = y_series.value_counts(normalize=True)
        sorted_v_counts = v_counts.sort_index()
        return np.asarray(sorted_v_counts)

class PNDBUfesDataset(MyDataset):
    def __init__(self, root: str, csv_train: str, csv_test: str, train: bool = True):
        super(PNDBUfesDataset, self).__init__(root, csv_train, csv_test, train)
        self.transform_center = transforms.Compose([
            CropCenterSquare(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.x = "path"
        self.y = "lesion"
        self.max_length = len("without displasya")

    def __getitem__(self, index) -> Dict[str, Any]:
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')

        item_dict = {"image": img, **self.df.loc[index].to_dict()}
        return item_dict
