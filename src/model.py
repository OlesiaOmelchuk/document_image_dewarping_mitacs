import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------
# Model Architecture (Transformer+U-Net)
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, p=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=False)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), dim)
        )
    def forward(self, x):  # x: [HW,B,D]
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x

class MultiStageTransformerEncoder(nn.Module):
    def __init__(self, img_channels=3, embed_dims=[64,128,256], patch_sizes=[8,16,2], depths=[2,2,2], heads=[2,4,8]):
        super().__init__()
        self.stages = nn.ModuleList()
        self.embed_dims = embed_dims
        for i, d in enumerate(embed_dims):
            in_ch = img_channels if i == 0 else embed_dims[i-1]
            self.stages.append(nn.ModuleDict({
                "proj": nn.Conv2d(in_ch, d, kernel_size=patch_sizes[i], stride=patch_sizes[i]),
                "blocks": nn.ModuleList([TransformerBlock(d, heads[i]) for _ in range(depths[i])])
            }))
    def forward(self, x):
        skips = []
        for s in self.stages:
            x = s["proj"](x)             # [B,D,h,w]
            B, D, h, w = x.shape
            x_seq = rearrange(x, "b d h w -> (h w) b d")
            for blk in s["blocks"]:
                x_seq = blk(x_seq)
            x = rearrange(x_seq, "(h w) b d -> b d h w", h=h, w=w)
            skips.append(x)
        return skips  # [low-res ... high-res]

class UNetDecoder(nn.Module):
    def __init__(self, embed_dims=[64,128,256], out_ch=2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dims[2], embed_dims[1], 2, 2)
        self.c1  = nn.Sequential(nn.Conv2d(embed_dims[1]*2, embed_dims[1], 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(embed_dims[1], embed_dims[1], 3, padding=1), nn.ReLU(True))
        self.up2 = nn.ConvTranspose2d(embed_dims[1], embed_dims[0], 2, 2)
        self.c2  = nn.Sequential(nn.Conv2d(embed_dims[0]*2, embed_dims[0], 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1), nn.ReLU(True))
        self.up3 = nn.ConvTranspose2d(embed_dims[0], embed_dims[0]//2, 2, 2)
        self.c3  = nn.Sequential(nn.Conv2d(embed_dims[0]//2, embed_dims[0]//2, 3, padding=1), nn.ReLU(True))
        self.out = nn.Conv2d(embed_dims[0]//2, out_ch, 1)
    def forward(self, skips):
        x = skips[-1]
        x = self.up1(x)
        s1 = F.interpolate(skips[1], size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s1], dim=1); x = self.c1(x)

        x = self.up2(x)
        s0 = F.interpolate(skips[0], size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s0], dim=1); x = self.c2(x)

        x = self.up3(x); x = self.c3(x)
        return self.out(x)

class FlowGenerator(nn.Module):
    """Predicts flow to transform input between domains."""
    def __init__(self, img_channels=3, max_disp=48.0):
        super().__init__()
        self.enc = MultiStageTransformerEncoder(img_channels=img_channels)
        self.dec = UNetDecoder()
        self.max_disp = max_disp
    def forward(self, x):
        B, C, H, W = x.shape
        skips = self.enc(x)
        flow = self.dec(skips)
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        # constrain displacement magnitude for stability
        # flow = torch.tanh(flow) * self.max_disp
        flow = flow * 10.0  # EXPERIMENT: 10x amplification for visibility
        return flow