import sys

sys.path.append("../base-trainer")

import torch

from light_infer.Module.trainer import Trainer


def demo():
    batch_size = 32
    accum_iter = 1
    num_workers = 16
    model_file_path = "./output/test/model_last.pth"
    model_file_path = None
    weights_only = False
    device = "auto"
    dtype = torch.float32
    warm_step_num = 0
    finetune_step_num = -1
    lr = 2e-4
    lr_batch_size = 256
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = "Loss"
    is_metric_lower_better = True
    sample_results_freq = 1
    use_amp = False
    quick_test = False

    trainer = Trainer(
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

    trainer.train()
    return True
