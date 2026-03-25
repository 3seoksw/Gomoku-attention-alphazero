import torch
import torch.nn as nn
from env.patterns import build_masks
from models.base_model import BaseModel
from models.policy_value_model import Backbone, PolicyValueHeads


class PatchEmbedding(nn.Module):
    def __init__(self, n_dim: int = 128, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
        self.patchify = nn.Unfold(patch_size, stride=1)
        # self.proj = nn.Linear(n_dim * patch_size * patch_size, n_dim)
        self.proj = nn.Sequential(
            nn.Linear(n_dim * patch_size * patch_size, n_dim * 2),
            nn.ReLU(),
            nn.Linear(n_dim * 2, n_dim),
        )

    def forward(self, x):
        # x: [B, 128, 9, 9]
        patches = self.patchify(x).transpose(1, 2)  # [B, patch^2, patch^2 * n_dim]
        proj = self.proj(patches)  # [B, patch^2, n_dim]
        return proj


class PatternCrossAttn(nn.Module):
    def __init__(
        self,
        n_dim: int = 128,
        n_in_a_row: int = 5,
        n_heads: int = 4,
    ):
        super().__init__()
        masks, self.mask_names = build_masks(n_in_a_row)  # [28, 25]
        self.register_buffer("masks", masks)
        self.n_patterns, self.n_patches = masks.shape[0], masks.shape[1]
        self.n_dim = n_dim

        # Query
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, n_dim))
        self.query_proj = nn.Identity()

        # Key & Value
        self.pattern_proj = nn.Sequential(
            nn.Linear(self.n_patches, n_dim),
            # nn.ReLU(),
            # nn.Linear(n_dim, n_dim),
        )
        self.pattern_weights = nn.Parameter(torch.ones(self.n_patterns))

        # Cross Attention
        self.mha = nn.MultiheadAttention(
            n_dim,
            n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(n_dim)
        self.linear = nn.Sequential(
            nn.Linear(n_dim, n_dim * 2),
            nn.ReLU(),
            nn.Linear(n_dim * 2, n_dim),
        )
        self.norm2 = nn.LayerNorm(n_dim)

    def forward(self, x: torch.Tensor):
        # x: patches [B, 25, 25 * 128]
        batch_size = x.shape[0]
        query = self.query_proj(x)  # [B, 25, 128]
        query = query + self.pos_embed

        kv = self.pattern_proj(self.masks)  # [28, 128]
        kv = kv.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 28, 128]
        # kv = kv * self.pattern_weights.unsqueeze(-1)

        # attn_weights.shape  [B, 25, 28]
        mha, attn_weights = self.mha(query=query, key=kv, value=kv)
        mha = self.norm(query + mha)  # [B, 25, 128]
        mha = self.norm2(mha + self.linear(mha))
        return mha, attn_weights


class AttnPolicyValue(BaseModel):
    def __init__(
        self,
        board_size: int = 9,
        n_in_a_row: int = 5,
        n_channels: int = 4,
        n_dim: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
    ):
        super().__init__(board_size, n_channels)
        self.n_dim = n_dim
        self.n = n_in_a_row

        self.backbone = Backbone(n_channels, n_dim, n_blocks)

        self.patch_emb = PatchEmbedding(n_dim, n_in_a_row)
        self.pattern_xattn = PatternCrossAttn(n_dim, n_in_a_row, n_heads)

        self.policy_value = PolicyValueHeads(board_size, n_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res_blocks = self.backbone(x)

        patches = self.patch_emb(res_blocks)
        mha, attn_weights = self.pattern_xattn(patches)

        feat = mha.view(-1, self.n, self.n, self.n_dim).permute(0, 3, 1, 2)
        feat = nn.functional.interpolate(
            feat, size=(9, 9), mode="bilinear", align_corners=False
        )

        policy, value = self.policy_value(feat)
        return policy, value, attn_weights


if __name__ == "__main__":
    x = torch.randn((64, 128, 9, 9), dtype=torch.float32)
    model = PatchEmbedding()
    xattn = PatternCrossAttn()
    model(x)
