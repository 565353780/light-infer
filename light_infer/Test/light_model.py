import os
from torch.utils.data import DataLoader


from light_infer.Dataset.light import LightDataset
from light_infer.Model.light import LightModel


def test():
    home = os.environ["HOME"]
    root_dataset_folder_path = home + "/chLi/Dataset/Light/"

    dataset = LightDataset(root_dataset_folder_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=1,
    )

    model = LightModel()

    for data in dataloader:
        result_dict = model(data)

        g = result_dict["g"]
        print(g.shape)

        break
    return True
