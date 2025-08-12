import torch
from torch import nn

from light_infer.Model.point_embed import PointEmbed


class LightModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        dropout_p: float = 0.2,  # 新增 dropout 参数
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        self.ranliao_embedding = nn.Embedding(4, self.hidden_dim)
        self.ranliao_density_embedding = nn.Embedding(6, self.hidden_dim)
        self.ningjiao_density_embedding = nn.Embedding(5, self.hidden_dim)

        self.ningjiao_height_encoder = PointEmbed(1, 64, self.hidden_dim)
        self.add_angle_encoder = PointEmbed(1, 64, self.hidden_dim)
        self.bochang_encoder = PointEmbed(1, 64, self.hidden_dim)

        self.g_decoder = nn.Sequential(
            nn.Linear(6 * self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(128, 1),
        )
        return

    def forward(self, data: dict) -> dict:
        ranliao_idx = data["ranliao_idx"].squeeze(1)
        ranliao_density_idx = data["ranliao_density_idx"].squeeze(1)
        ningjiao_density_idx = data["ningjiao_density_idx"].squeeze(1)

        ningjiao_height = data["ningjiao_height"].unsqueeze(1)
        add_angle = data["add_angle"].unsqueeze(1)
        bochang = data["bochang"].unsqueeze(1)

        ranliao_embedding = self.ranliao_embedding(ranliao_idx)
        ranliao_density_embedding = self.ranliao_density_embedding(ranliao_density_idx)
        ningjiao_density_embedding = self.ningjiao_density_embedding(
            ningjiao_density_idx
        )

        ningjiao_height_feature = self.ningjiao_height_encoder(ningjiao_height).squeeze(
            1
        )
        add_angle_feature = self.add_angle_encoder(add_angle).squeeze(1)
        bochang_feature = self.bochang_encoder(bochang).squeeze(1)

        light_features = torch.stack(
            [
                ranliao_embedding,
                ranliao_density_embedding,
                ningjiao_density_embedding,
                ningjiao_height_feature,
                add_angle_feature,
                bochang_feature,
            ],
            dim=1,
        )

        g = self.g_decoder(light_features.view(light_features.shape[0], -1))

        result_dict = {
            "g": g,
        }

        return result_dict
