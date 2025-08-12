import torch.nn as nn


class SpectralCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)
        return

    def forward(self, data: dict) -> dict:
        x = data["input"]

        x = x.unsqueeze(1)  # (batch, 1, wavelength)
        x = self.conv(x).squeeze(-1)

        output = self.fc(x)

        result_dict = {
            "output": output,
        }

        return result_dict
