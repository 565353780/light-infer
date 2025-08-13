import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from light_infer.Dataset.light import LightDataset


def test():
    home = os.environ["HOME"]
    root_dataset_folder_path = home + "/chLi/Dataset/Light/"

    light_dataset = LightDataset(root_dataset_folder_path)
    dataloader = DataLoader(
        dataset=light_dataset,
        batch_size=256,
        num_workers=16,
    )

    g_min = float("inf")
    g_max = -float("inf")
    for data_dict in tqdm(dataloader):
        g = data_dict["g"]
        curr_g_min = torch.min(g).item()
        curr_g_max = torch.max(g).item()
        g_min = min(g_min, curr_g_min)
        g_max = max(g_max, curr_g_max)

    print("g in : [", g_min, ",", g_max, "]")
    return True
