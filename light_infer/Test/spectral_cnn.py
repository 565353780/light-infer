import torch

from light_infer.Model.spectral_cnn import SpectralCNN


def test():
    data_dict = {
        "input": torch.randn(16, 1),
    }
    spectral_cnn = SpectralCNN()

    result_dict = spectral_cnn(data_dict)

    output = result_dict["output"]
    print(output.shape)
    return True
