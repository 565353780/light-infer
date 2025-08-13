import os
import torch
import numpy as np
from tqdm import trange
from typing import Union
from shutil import rmtree

from light_infer.Method.path import removeFile, renameFile
from light_infer.Method.render import renderInferXYData
from light_infer.Method.video import toVideo
from light_infer.Module.detector import Detector


def linspaceByStep(data_min: float, data_max: float, data_step: float) -> np.ndarray:
    data_num = int(np.floor((data_max - data_min) / data_step)) + 1
    data_array = np.linspace(data_min, data_max, data_num)
    return data_array


class ModelViewer(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        dtype=torch.float64,
        device: str = "cpu",
        use_ema: bool = True,
    ) -> None:
        self.detector = Detector(model_file_path, dtype, device, use_ema)
        return

    def loadModel(self, model_file_path: str) -> bool:
        return self.detector.loadModel(model_file_path)

    def toVideoWithNingjiaoHeight(
        self,
        ranliao_idx: int,
        ranliao_density_idx: int,
        ningjiao_density_idx: int,
        add_angle: float,
        save_video_file_path: str,
        ningjiao_height_min: float = 0.2,
        ningjiao_height_max: float = 1.5,
        ningjiao_height_step: float = 0.01,
        bochang_min: float = 400.0,
        bochang_max: float = 800.0,
        bochang_step: float = 0.5,
        overwrite: bool = False,
    ) -> bool:
        if os.path.exists(save_video_file_path):
            if not overwrite:
                return True

            removeFile(save_video_file_path)

        tmp_video_image_folder_path = "./tmp_video_images/"
        if os.path.exists(tmp_video_image_folder_path):
            rmtree(tmp_video_image_folder_path)

        os.makedirs(tmp_video_image_folder_path)

        ningjiao_height_array = linspaceByStep(
            ningjiao_height_min, ningjiao_height_max, ningjiao_height_step
        )

        bochang = linspaceByStep(bochang_min, bochang_max, bochang_step)

        print("[INFO][ModelViewer::toVideoWithNingjiaoHeight]")
        print("\t start render images...")
        for i in trange(ningjiao_height_array.shape[0]):
            ningjiao_height = ningjiao_height_array[i]

            pred_g = self.detector.detectWithBochang(
                ranliao_idx,
                ranliao_density_idx,
                ningjiao_density_idx,
                ningjiao_height,
                add_angle,
                bochang,
            )

            renderInferXYData(
                bochang,
                pred_g,
                render=False,
                save_image_file_path=tmp_video_image_folder_path + str(i) + ".jpg",
            )

        tmp_save_video_file_path = (
            save_video_file_path[:-4] + "_tmp" + save_video_file_path[-4:]
        )
        toVideo(
            tmp_video_image_folder_path,
            tmp_save_video_file_path,
        )

        renameFile(tmp_save_video_file_path, save_video_file_path)

        return True
