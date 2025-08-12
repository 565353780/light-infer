import os
import numpy as np
from typing import Union


def loadLightFile(light_file_path: str) -> Union[dict, None]:
    if not os.path.exists(light_file_path):
        print("[ERROR][io::loadLightFile]")
        print("\t light file not exist!")
        print("\t light_file_path:", light_file_path)
        return None

    with open(light_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    light_data_dict = {}

    xy_data_start_idx = -1
    xy_data_end_idx = -1
    for i in range(len(lines)):
        if "XYDATA" in lines[i]:
            xy_data_start_idx = i

        if xy_data_start_idx > 0:
            if lines[i] == "\n":
                xy_data_end_idx = i
                break

    if xy_data_start_idx == -1:
        print("[ERROR][io::loadLightFile]")
        print("\t XYDATA not found!")
        return None

    if xy_data_end_idx == -1:
        print("[ERROR][io::loadLightFile]")
        print("\t XYDATA length not found!")
        return None

    header_lines = lines[:xy_data_start_idx]
    xy_data_lines = lines[xy_data_start_idx + 1 : xy_data_end_idx]

    for header in header_lines:
        header_data_list = header.split("\n")[0].split()
        if len(header_data_list) < 2:
            continue

        header_key = header_data_list[0]

        header_value = ""
        for header_data in header_data_list[1:]:
            header_value += header_data + " "

        header_value = header_value[:-1]

        light_data_dict[header_key] = header_value

    xy_data_array = []
    for xy_data in xy_data_lines:
        xy_data_list = xy_data.split("\n")[0].split()
        float_xy_data = [float(str_data) for str_data in xy_data_list]

        xy_data_array.append(float_xy_data)

    xy_data_array = np.asarray(xy_data_array, dtype=np.double)

    light_data_dict["XYDATA"] = xy_data_array

    return light_data_dict
