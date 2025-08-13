import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from light_infer.Method.path import createFileFolder, renameFile


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


def renderBatchInferXYData(
    batch_bochang: np.ndarray,
    batch_pred_g: np.ndarray,
    batch_gt_g: Union[np.ndarray, None] = None,
    render: bool = True,
    save_image_file_path: Union[str, None] = None,
) -> bool:
    image_width = 1920
    image_height = 1080
    dpi = 100

    batch_size = batch_bochang.shape[0]

    col_num = int(np.ceil(np.sqrt(batch_size)))
    row_num = batch_size // col_num + (batch_size % col_num > 0)

    fig, axes = plt.subplots(
        row_num,
        col_num,
        figsize=(1.0 * image_width / dpi, 1.0 * image_height / dpi),
        dpi=dpi,
    )

    axes = np.atleast_2d(axes)

    for i in range(batch_size):
        row_idx = i // col_num
        col_idx = i % col_num

        ax = axes[row_idx, col_idx]

        ax.plot(
            batch_bochang[i],
            batch_pred_g[i],
            label="Pred g",
            color="red",
            linewidth=2,
        )

        if batch_gt_g is not None:
            ax.plot(
                batch_bochang[i],
                batch_gt_g[i],
                label="GT g",
                color="blue",
                linewidth=2,
                linestyle="--",
            )

        ax.set_title("No." + str(i))
        ax.set_xlabel("wavelength")
        ax.set_ylabel("g")
        ax.set_title("g=Model(wavelength)")
        ax.set_xlim(400, 800)
        ax.set_ylim(-0.5, 0.8)

    plt.tight_layout()

    if render:
        plt.show()

    if save_image_file_path is not None:
        tmp_save_image_file_path = (
            save_image_file_path[:-4] + "_tmp" + save_image_file_path[-4:]
        )
        createFileFolder(tmp_save_image_file_path)

        fig.savefig(tmp_save_image_file_path, dpi=dpi)

        renameFile(tmp_save_image_file_path, save_image_file_path)

    plt.close()
    return True


def renderInferXYData(
    bochang: np.ndarray,
    pred_g: np.ndarray,
    gt_g: Union[np.ndarray, None] = None,
    render: bool = True,
    save_image_file_path: Union[str, None] = None,
) -> bool:
    if gt_g is not None:
        batch_gt_g = gt_g[np.newaxis, :]
    else:
        batch_gt_g = gt_g

    return renderBatchInferXYData(
        bochang[np.newaxis, :],
        pred_g[np.newaxis, :],
        batch_gt_g,
        render,
        save_image_file_path,
    )
