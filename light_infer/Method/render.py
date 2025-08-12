import numpy as np
import matplotlib.pyplot as plt
from torch import nn


def renderXYData(xy_data_array: np.ndarray) -> bool:
    wavelength = xy_data_array[:, 0]
    y1 = xy_data_array[:, 1]  # g(lum)
    y2 = xy_data_array[:, 2]  # Delta I
    y3 = xy_data_array[:, 3]  # CD [mdeg]
    y4 = xy_data_array[:, 4]  # DC [V]

    # ---------- 3. 画图 ----------
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体，支持中文
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CPL 光谱数据", fontsize=16)

    ax[0, 0].plot(wavelength, y1, color="tab:blue")
    ax[0, 0].set_title("g(lum)")
    ax[0, 0].set_xlabel("波长 [nm]")
    ax[0, 0].set_ylabel("g(lum)")

    ax[0, 1].plot(wavelength, y2, color="tab:orange")
    ax[0, 1].set_title("ΔI")
    ax[0, 1].set_xlabel("波长 [nm]")
    ax[0, 1].set_ylabel("ΔI")

    ax[1, 0].plot(wavelength, y3, color="tab:green")
    ax[1, 0].set_title("CD [mdeg]")
    ax[1, 0].set_xlabel("波长 [nm]")
    ax[1, 0].set_ylabel("CD [mdeg]")

    ax[1, 1].plot(wavelength, y4, color="tab:red")
    ax[1, 1].set_title("DC [V]")
    ax[1, 1].set_xlabel("波长 [nm]")
    ax[1, 1].set_ylabel("DC [V]")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    return True


def renderInferXYData(xy_data_array: np.ndarray, model: nn.Module) -> np.ndarray:
    wavelength = xy_data_array[:, 0]
    gt_y1 = xy_data_array[:, 1]  # g(lum)

    infer_y1 = []

    return
