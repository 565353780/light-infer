import os
import torch
import numpy as np
from tqdm import trange

from light_infer.Module.detector import Detector
from light_infer.Dataset.light import LightDataset
from light_infer.Method.render import renderBatchInferXYData


def demo():
    model_file_path = "./output/20250812_19:01:24/model_best.pth"
    dtype = torch.float64
    device = "cuda:0"
    use_ema = True

    detector = Detector(model_file_path, dtype, device, use_ema)

    batch_bochang = []
    batch_gt_g = []
    batch_pred_g = []

    dataset = LightDataset(
        os.environ["HOME"] + "/chLi/Dataset/Light/", load_full_xy_data=True
    )
    print("start collect predict g...")
    for i in trange(9):
        data = dataset[i]
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

        batch_bochang.append(bochang)
        batch_gt_g.append(gt_g)
        batch_pred_g.append(pred_g)

    batch_bochang = np.stack(batch_bochang, axis=0)
    batch_gt_g = np.stack(batch_gt_g, axis=0)
    batch_pred_g = np.stack(batch_pred_g, axis=0)

    renderBatchInferXYData(batch_bochang, batch_pred_g, batch_gt_g)

    return True
