import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Union

from light_infer.Model.light import LightModel


def toTensor(
    data: Union[torch.Tensor, np.ndarray],
    dtype=torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    data = data.to(device, dtype=dtype)
    return data


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        dtype=torch.float64,
        device: str = "cpu",
        use_ema: bool = True,
    ) -> None:
        self.dtype = dtype
        self.device = device

        self.use_ema = use_ema

        self.model = LightModel().to(self.device, dtype=self.dtype)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][CFMSampler::loadModel]")
            print("\t model_file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(
            model_file_path, map_location=torch.device(self.device), weights_only=False
        )

        if self.use_ema:
            self.model.load_state_dict(model_dict["ema_model"])
        else:
            self.model.load_state_dict(model_dict["model"])

        self.model.eval()

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def detect(
        self,
        ranliao_idx: Union[torch.Tensor, np.ndarray],
        ranliao_density_idx: Union[torch.Tensor, np.ndarray],
        ningjiao_density_idx: Union[torch.Tensor, np.ndarray],
        ningjiao_height: Union[torch.Tensor, np.ndarray],
        add_angle: Union[torch.Tensor, np.ndarray],
        bochang: Union[torch.Tensor, np.ndarray],
    ) -> dict:
        if isinstance(ranliao_idx, np.ndarray):
            ranliao_idx = torch.from_numpy(ranliao_idx)

        ranliao_idx = toTensor(ranliao_idx, torch.int64, self.device)
        ranliao_density_idx = toTensor(ranliao_density_idx, torch.int64, self.device)
        ningjiao_density_idx = toTensor(ningjiao_density_idx, torch.int64, self.device)
        ningjiao_height = toTensor(ningjiao_height, self.dtype, self.device)
        add_angle = toTensor(add_angle, self.dtype, self.device)
        bochang = toTensor(bochang, self.dtype, self.device)

        data_dict = {
            "ranliao_idx": ranliao_idx,
            "ranliao_density_idx": ranliao_density_idx,
            "ningjiao_density_idx": ningjiao_density_idx,
            "ningjiao_height": ningjiao_height,
            "add_angle": add_angle,
            "bochang": bochang,
        }

        result_dict = self.model(data_dict)

        return result_dict

    @torch.no_grad()
    def detectWithBatch(
        self,
        ranliao_idx: Union[torch.Tensor, np.ndarray],
        ranliao_density_idx: Union[torch.Tensor, np.ndarray],
        ningjiao_density_idx: Union[torch.Tensor, np.ndarray],
        ningjiao_height: Union[torch.Tensor, np.ndarray],
        add_angle: Union[torch.Tensor, np.ndarray],
        bochang: Union[torch.Tensor, np.ndarray],
        batch_size: int = 400,
    ) -> dict:
        combined_result = None

        num_batches = (ranliao_idx.shape[0] + batch_size - 1) // batch_size
        print("[INFO][Detector::detectWithBatch]")
        print("\t start detect with batch, num_batches:", num_batches)
        for i in trange(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, ranliao_idx.shape[0])

            batch_ranliao_idx = ranliao_idx[start_idx:end_idx]
            batch_ranliao_density_idx = ranliao_density_idx[start_idx:end_idx]
            batch_ningjiao_density_idx = ningjiao_density_idx[start_idx:end_idx]
            batch_ningjiao_height = ningjiao_height[start_idx:end_idx]
            batch_add_angle = add_angle[start_idx:end_idx]
            batch_bochang = bochang[start_idx:end_idx]

            batch_result = self.detect(
                batch_ranliao_idx,
                batch_ranliao_density_idx,
                batch_ningjiao_density_idx,
                batch_ningjiao_height,
                batch_add_angle,
                batch_bochang,
            )

            if combined_result is None:
                combined_result = {k: [] for k in batch_result.keys()}

            for k, v in batch_result.items():
                combined_result[k].append(v)

        final_result = {}
        for k, v_list in combined_result.items():
            if isinstance(v_list[0], torch.Tensor):
                final_result[k] = torch.cat(v_list, dim=0)
            else:
                final_result[k] = v_list

        return final_result

    @torch.no_grad()
    def renderInferXYData(
        self,
        ranliao_idx: int,
        ranliao_density_idx: int,
        ningjiao_density_idx: int,
        ningjiao_height: float,
        add_angle: float,
        xy_data_array: np.ndarray,
    ) -> bool:
        batch_size = xy_data_array.shape[0]

        wavelength = xy_data_array[:, 0]
        gt_g = xy_data_array[:, 1]  # g(lum)

        data_dict = {
            "ranliao_idx": torch.tensor(
                [ranliao_idx], dtype=torch.int64, device=self.device
            )
            .view(1, 1)
            .expand(batch_size, 1),
            "ranliao_density_idx": torch.tensor(
                [ranliao_density_idx], dtype=torch.int64, device=self.device
            )
            .view(1, 1)
            .expand(batch_size, 1),
            "ningjiao_density_idx": torch.tensor(
                [ningjiao_density_idx], dtype=torch.int64, device=self.device
            )
            .view(1, 1)
            .expand(batch_size, 1),
            "ningjiao_height": torch.tensor(
                [ningjiao_height], dtype=self.dtype, device=self.device
            )
            .view(1, 1)
            .expand(batch_size, 1),
            "add_angle": torch.tensor([add_angle], dtype=self.dtype, device=self.device)
            .view(1, 1)
            .expand(batch_size, 1),
            "bochang": torch.from_numpy(wavelength)
            .view(-1, 1)
            .to(dtype=self.dtype, device=self.device),
        }

        result_dict = self.model(data_dict)

        pred_g = result_dict["g"].cpu().numpy().reshape(-1)

        # ---------- 画图 ----------
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体，支持中文
        plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("CPL 光谱数据", fontsize=16)

        ax[0].plot(wavelength, gt_g, color="tab:blue")
        ax[0].set_title("GT g(lum)")
        ax[0].set_xlabel("波长 [nm]")
        ax[0].set_ylabel("GT g(lum)")

        ax[1].plot(wavelength, pred_g, color="tab:orange")
        ax[1].set_title("Pred g(lum)")
        ax[1].set_xlabel("波长 [nm]")
        ax[1].set_ylabel("Pred g(lum)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

        return True
