import torch
import torch.nn as nn
from env.patterns import build_masks
from models.base_model import BaseModel
from models.policy_value_model import PolicyValueHeads


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, shared_proj: nn.Linear):
        super().__init__()
        self.patch_size = patch_size
        self.patchify = nn.Unfold(patch_size, stride=1)
        self.shared_proj = shared_proj  # Linear(n_patches, n_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, n_channels, 9, 9]
        batch_size = x.shape[0]
        n_patches = self.patch_size**2

        patches = self.patchify(x).transpose(1, 2)  # [B, 25, n_channels * 25]
        patches = patches.view(
            batch_size, -1, x.shape[1], n_patches
        )  # [B, 25, n_channels, 25]
        patches = patches.mean(dim=2)  # [B, 25, 25] — avg over channels
        return self.shared_proj(patches)  # [B, 25, n_dim]


class PatternCrossAttn(nn.Module):
    def __init__(
        self,
        shared_proj: nn.Linear,
        n_dim: int = 128,
        n_in_a_row: int = 5,
        n_heads: int = 4,
    ):
        super().__init__()
        masks, self.mask_names = build_masks(n_in_a_row)  # [28, 25]
        self.register_buffer("masks", masks)
        self.n_patterns, self.n_patches = masks.shape[0], masks.shape[1]
        self.n_dim = n_dim

        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, n_dim))
        self.shared_proj = (
            shared_proj  # same Linear(n_patches, n_dim) as PatchEmbedding
        )

        # Cross Attention
        self.mha = nn.MultiheadAttention(n_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(n_dim)
        self.linear = nn.Sequential(
            nn.Linear(n_dim, n_dim * 2),
            nn.ReLU(),
            nn.Linear(n_dim * 2, n_dim),
        )
        self.norm2 = nn.LayerNorm(n_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, 25, n_dim] from PatchEmbedding
        batch_size = x.shape[0]

        query = x + self.pos_embed  # [B, 25, n_dim]
        kv = self.shared_proj(self.masks)  # [28, n_dim]
        kv = kv.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 28, n_dim]

        # attn_weights: [B, 25, 28]
        mha_out, attn_weights = self.mha(query=query, key=kv, value=kv)
        mha_out = self.norm(query + mha_out)
        mha_out = self.norm2(mha_out + self.linear(mha_out))
        return mha_out, attn_weights


class AttnPolicyValue(BaseModel):
    def __init__(
        self,
        board_size: int = 9,
        n_in_a_row: int = 5,
        n_channels: int = 4,
        n_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__(board_size, n_channels)
        self.n_dim = n_dim
        self.n = n_in_a_row

        n_patches = n_in_a_row**2
        shared_proj = nn.Linear(n_patches, n_dim)

        self.patch_emb = PatchEmbedding(n_in_a_row, shared_proj)
        self.pattern_xattn = PatternCrossAttn(shared_proj, n_dim, n_in_a_row, n_heads)
        self.policy_value = PolicyValueHeads(board_size, n_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self.patch_emb(x)  # [B, 25, n_dim]
        mha, attn_weights = self.pattern_xattn(patches)  # [B, 25, n_dim]

        feat = mha.view(-1, self.n, self.n, self.n_dim).permute(0, 3, 1, 2)
        feat = nn.functional.interpolate(
            feat,
            size=(self.board_size, self.board_size),
            mode="bilinear",
            align_corners=False,
        )

        policy, value = self.policy_value(feat)
        return policy, value, attn_weights


if __name__ == "__main__":
    model = AttnPolicyValue()
    x = torch.randn(4, 4, 9, 9)
    policy, value, attn = model(x)
    print("policy:", policy.shape)
    print("value:", value.shape)
    print("attn:", attn.shape)
