import os
import torch

from light_infer.Module.detector import Detector
from light_infer.Dataset.light import LightDataset
from light_infer.Method.render import renderInferXYData


def demo():
    model_file_path = "./output/20250812_19:01:24/model_best.pth"
    dtype = torch.float64
    device = "cuda:0"
    use_ema = True

    detector = Detector(model_file_path, dtype, device, use_ema)

    dataset = LightDataset(
        os.environ["HOME"] + "/chLi/Dataset/Light/", load_full_xy_data=True
    )
    data = dataset[0]
    ranliao_idx = data["ranliao_idx"][0].item()
    ranliao_density_idx = data["ranliao_density_idx"][0].item()
    ningjiao_density_idx = data["ningjiao_density_idx"][0].item()
    ningjiao_height = data["ningjiao_height"][0].item()
    add_angle = data["add_angle"][0].item()
    xy_data = data["xy_data"].numpy()
    bochang = xy_data[:, 0]
    gt_g = xy_data[:, 1]

    pred_g = detector.detectWithBochang(
        ranliao_idx,
        ranliao_density_idx,
        ningjiao_density_idx,
        ningjiao_height,
        add_angle,
        bochang,
    )

    renderInferXYData(bochang, gt_g, pred_g)

    return True
