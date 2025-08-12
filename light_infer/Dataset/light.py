import torch
import random
import numpy as np
from torch.utils.data import Dataset

from light_infer.Config.ranliao import RANLIAO_IDXS
from light_infer.Method.io import loadLightFile
from light_infer.Method.dataset_io import loadLightDataset


class LightDataset(Dataset):
    def __init__(
        self,
        root_dataset_folder_path: str,
        split: str = "train",
        dtype=torch.float32,
        load_full_xy_data: bool = False,
    ) -> None:
        self.split = split
        self.dtype = dtype
        self.load_full_xy_data = load_full_xy_data

        self.light_data_dict = loadLightDataset(root_dataset_folder_path)
        assert isinstance(self.light_data_dict, dict)

        self.data_list = list(self.light_data_dict.keys())
        return

    def __len__(self):
        if self.split == "train":
            return 10 * len(self.data_list)
        else:
            return len(self.data_list)

    def __getitem__(self, index):
        index = index % len(self.data_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        light_file_path = self.data_list[index]

        light_info_dict = self.light_data_dict[light_file_path]

        light_record_data_dict = loadLightFile(light_file_path)

        xy_data = light_record_data_dict["XYDATA"]

        train_idx = random.randint(0, xy_data.shape[0] - 1)

        data = {
            "ranliao_idx": torch.tensor(
                [RANLIAO_IDXS[light_info_dict["ranliao"]]], dtype=torch.int64
            ),
            "ranliao_density_idx": torch.tensor(
                [light_info_dict["ranliao_density_idx"] - 1], dtype=torch.int64
            ),
            "ningjiao_density_idx": torch.tensor(
                [light_info_dict["ningjiao_density_idx"] - 1], dtype=torch.int64
            ),
            "ningjiao_height": torch.tensor(
                [light_info_dict["ningjiao_height"]], dtype=self.dtype
            ),
            "add_angle": torch.tensor([light_info_dict["add_angle"]], dtype=self.dtype),
            "bochang": torch.tensor([xy_data[train_idx][0]], dtype=self.dtype),
            "g": torch.tensor([xy_data[train_idx][1]], dtype=self.dtype),
        }

        if self.load_full_xy_data:
            data["xy_data"] = torch.tensor(xy_data, dtype=self.dtype)

        return data
