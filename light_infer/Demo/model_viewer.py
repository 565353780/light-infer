import os
import torch

from light_infer.Dataset.light import LightDataset
from light_infer.Module.model_viewer import ModelViewer


def demo():
    model_file_path = "./output/20250812_19:01:24/model_best.pth"
    dtype = torch.float64
    device = "cuda:0"
    use_ema = True
    save_video_file_path = "./output/demo_video.mp4"
    overwrite = True

    model_viewer = ModelViewer(model_file_path, dtype, device, use_ema)

    dataset = LightDataset(
        os.environ["HOME"] + "/chLi/Dataset/Light/", load_full_xy_data=True
    )
    data = dataset[0]
    ranliao_idx = data["ranliao_idx"][0].item()
    ranliao_density_idx = data["ranliao_density_idx"][0].item()
    ningjiao_density_idx = data["ningjiao_density_idx"][0].item()
    ningjiao_height = data["ningjiao_height"][0].item()
    add_angle = data["add_angle"][0].item()

    model_viewer.toVideoWithNingjiaoHeight(
        ranliao_idx,
        ranliao_density_idx,
        ningjiao_density_idx,
        add_angle,
        save_video_file_path,
        ningjiao_height_min=0.2,
        ningjiao_height_max=1.5,
        ningjiao_height_step=0.001,
        bochang_min=400.0,
        bochang_max=800.0,
        bochang_step=0.5,
        overwrite=overwrite,
    )
    return True
