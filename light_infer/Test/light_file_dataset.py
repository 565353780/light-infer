import os

from light_infer.Dataset.light_file import LightFileDataset


def test():
    home = os.environ["HOME"]
    light_file_path = home + "/chLi/Dataset/Light/3+.txt"

    light_file_dataset = LightFileDataset(light_file_path)

    data_dict = light_file_dataset[0]

    print(data_dict["input"].shape)
    print(data_dict["output"].shape)
    return True
