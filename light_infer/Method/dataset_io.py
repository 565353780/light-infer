import os
from typing import Union


def loadMainDataset(main_dataset_folder_path: str) -> Union[dict, None]:
    if not os.path.exists(main_dataset_folder_path):
        print("[ERROR][dataset_io::loadMainDataset]")
        print("\t main dataset folder not exist!")
        print("\t main_dataset_folder_path:", main_dataset_folder_path)
        return None

    dataset_dict = {}

    category_list = os.listdir(main_dataset_folder_path)

    valid_category_list = []

    for category in category_list:
        if category[0] == ".":
            continue
        valid_category_list.append(category)

    for category in valid_category_list:
        sub_folder_path = main_dataset_folder_path + category + "/"

        first_info_list = os.listdir(sub_folder_path)

        for first_info in first_info_list:
            if category not in first_info:
                continue

            valid_first_info_list = first_info[len(category) + 1 :].split("号")

            ningjiao_density_idx = int(valid_first_info_list[0])
            ningjiao_height = float(valid_first_info_list[1][:-2])

            subsub_folder_path = sub_folder_path + first_info + "/"

            data_files = os.listdir(subsub_folder_path)

            for data_file in data_files:
                if not data_file.endswith(".txt"):
                    continue

                ranliao_density_idx = int(data_file[0])
                add_angle = 45.0 if data_file[1] == "+" else -45.0

                data_file_path = subsub_folder_path + data_file

                dataset_dict[data_file_path] = {
                    "ranliao": category,
                    "ningjiao_density_idx": ningjiao_density_idx,
                    "ningjiao_height": ningjiao_height,
                    "ranliao_density_idx": ranliao_density_idx,
                    "add_angle": add_angle,
                }

    return dataset_dict


def loadAngleDataset(angle_dataset_folder_path: str) -> Union[dict, None]:
    if not os.path.exists(angle_dataset_folder_path):
        print("[ERROR][dataset_io::loadAngleDataset]")
        print("\t angle dataset folder not exist!")
        print("\t angle_dataset_folder_path:", angle_dataset_folder_path)
        return None

    dataset_dict = {}

    first_info_list = os.listdir(angle_dataset_folder_path)
    for first_info in first_info_list:
        sub_folder_path = angle_dataset_folder_path + first_info + "/"

        if not os.path.exists(sub_folder_path):
            continue

        data_info_list = first_info.split("_")

        category = data_info_list[0]
        ranliao_density_idx = int(data_info_list[1])
        ningjiao_density_idx = int(data_info_list[2])
        ningjiao_height = float(data_info_list[3])

        data_files = os.listdir(sub_folder_path)

        for data_file in data_files:
            if not data_file.endswith(".txt"):
                continue

            add_angle = float(data_file[:-4])

            data_file_path = sub_folder_path + data_file

            dataset_dict[data_file_path] = {
                "ranliao": category,
                "ningjiao_density_idx": ningjiao_density_idx,
                "ningjiao_height": ningjiao_height,
                "ranliao_density_idx": ranliao_density_idx,
                "add_angle": add_angle,
            }
    return dataset_dict


def loadLightDataset(root_dataset_folder_path: str) -> Union[dict, None]:
    main_dataset_folder_path = root_dataset_folder_path + "机器学习/"
    angle_dataset_folder_path = root_dataset_folder_path + "角度/"

    dataset_dict = {}

    main_dataset_dict = loadMainDataset(main_dataset_folder_path)
    assert isinstance(main_dataset_dict, dict)
    dataset_dict.update(main_dataset_dict)

    angle_dataset_dict = loadAngleDataset(angle_dataset_folder_path)
    assert isinstance(angle_dataset_dict, dict)
    dataset_dict.update(angle_dataset_dict)

    return dataset_dict
