import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from light_infer.Dataset.light_file import LightFileDataset
from light_infer.Model.spectral_cnn import SpectralCNN


class Trainer(BaseTrainer):
    def __init__(
        self,
        batch_size: int = 16,
        accum_iter: int = 1,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_amp: bool = False,
        quick_test: bool = False,
    ) -> None:
        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_amp,
            quick_test,
        )

        self.loss_func = nn.MSELoss()
        return

    def createDatasets(self) -> bool:
        light_file_path = "/Users/chli/chLi/Dataset/Light/3+.txt"

        self.dataloader_dict["name"] = {
            "dataset": LightFileDataset(light_file_path, "train", self.dtype),
            "repeat_num": 1,
        }

        self.dataloader_dict["eval"] = {
            "dataset": LightFileDataset(light_file_path, "val", self.dtype),
        }

        # crop data num for faster evaluation
        self.dataloader_dict["eval"]["dataset"].data_list = self.dataloader_dict[
            "eval"
        ]["dataset"].data_list[:64]
        return True

    def createModel(self) -> bool:
        self.model = SpectralCNN().to(self.device, dtype=self.dtype)
        return True

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if is_training:
            data_dict["drop_prob"] = 0.0
        else:
            data_dict["drop_prob"] = 0.0

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        gt_output = data_dict["output"]
        output = result_dict["output"]

        loss = self.loss_func(gt_output, output)

        loss_dict = {
            "Loss": loss,
        }

        return loss_dict
