import torch
import numpy as np
from torch.utils.data import Dataset

from light_infer.Method.io import loadLightFile


class LightFileDataset(Dataset):
    def __init__(
        self,
        light_file_path: str,
        split: str = "train",
        dtype=torch.float32,
    ) -> None:
        self.split = split
        self.dtype = dtype

        light_data_dict = loadLightFile(light_file_path)

        assert isinstance(light_data_dict, dict)

        self.xy_data = torch.from_numpy(light_data_dict["XYDATA"])
        return

    def __len__(self):
        return len(self.xy_data)

    def __getitem__(self, index):
        index = index % len(self.xy_data)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        xy_data = self.xy_data[index]

        data = {
            "input": xy_data[0].to(self.dtype),
            "output": xy_data[1].to(self.dtype),
        }

        return data
