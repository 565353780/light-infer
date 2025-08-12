import os

from light_infer.Dataset.light import LightDataset


def test():
    home = os.environ["HOME"]
    root_dataset_folder_path = home + "/chLi/Dataset/Light/"

    light_dataset = LightDataset(root_dataset_folder_path)

    data_dict = light_dataset[0]

    print(data_dict)
    return True
