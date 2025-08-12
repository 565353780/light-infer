from light_infer.Method.io import loadLightFile
from light_infer.Method.render import renderXYData


def demo():
    data_file_path = "/Users/chli/chLi/Dataset/Light/3+.txt"

    light_data_dict = loadLightFile(data_file_path)

    assert isinstance(light_data_dict, dict)

    renderXYData(light_data_dict["XYDATA"])
