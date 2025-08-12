from light_infer.Method.io import loadLightFile
from light_infer.Method.render import renderXYData


if __name__ == "__main__":
    data_file_path = "/Users/chli/Downloads/3+.txt"

    light_data_dict = loadLightFile(data_file_path)

    if light_data_dict is None:
        print("loadLightFile failed!")

    renderXYData(light_data_dict["XYDATA"])
