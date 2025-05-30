# models.py — **fully shape‑annotated**
# ==================================
# Every tensor creation / transform carries a `(batch, …)` comment
# so you can grep for “(” and quickly trace data sizes end‑to‑end.

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Helper: rasterise trajectories to an ego‑centred image tensor
# ------------------------------------------------------------

def rasterise(hist, grid_size: int = 128, cell_res: float = 0.5):
    """Convert the 50‑frame history in *batch* into a (B,3,H,W) raster.

    Input
    -----
    batch.x : (B, 50_agents, 50_timesteps, 6)
        Already ego‑centred at t = 49 ⇒ ego (agent‑0) at (0,0).
    batch.scale : (B,)  — metres ↔ normalised units scale factors.

    Output
    ------
    imgs : (B, 3, grid_size, grid_size)
        ch‑0  occupancy (0/1)
        ch‑1  time‑freshness ∈ [0,1]
        ch‑2  cos heading   ∈ [‑1,1]
    """
    B, A, T, _ = hist.shape  # (batch, agents, timesteps, 6)
    device = hist.device

    # (B, 3, H, W) — initialise to zeros
    imgs = torch.zeros((B, 3, grid_size, grid_size), device=device)

    # Pre‑compute constants --------------------------------------------------
    centre = (grid_size - 1) / 2.0  # pixel coord of (0,0)
    scale = 1.0 / cell_res  # metres → pixels

    # Split fields once for readability -------------------------------------
    xy = hist[..., :2]  # (B, A, T, 2)
    vxvy = hist[..., 2:4]  # (B, A, T, 2)

    # Pixel indices (long) ---------------------------------------------------
    x_pix = torch.round(xy[..., 0] * scale + centre).long()  # (B,A,T)
    y_pix = torch.round(xy[..., 1] * scale + centre).long()  # (B,A,T)

    in_bounds = (
            (x_pix >= 0) & (x_pix < grid_size) &
            (y_pix >= 0) & (y_pix < grid_size)
    )  # (B,A,T) bool

    # Pre‑compute timestep index tensor once (B,A,T) ------------------------
    t_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, A, T)

    # -----------------------------------------------------------------------
    for b in range(B):
        xp = x_pix[b][in_bounds[b]]  # (N,)
        yp = y_pix[b][in_bounds[b]]  # (N,)
        msk = in_bounds[b]  # same mask for t & θ

        # 0) OCCUPANCY -------------------------------------------------------
        imgs[b, 0].index_put_(
            (yp, xp),
            torch.ones_like(xp, dtype=imgs.dtype),
            accumulate=True,
        )

        # 1) MOST‑RECENT TIMESTEP -------------------------------------------
        imgs[b, 1].index_put_(
            (yp, xp),
            (t_idx[b][msk].float() / (T - 1)),  # normalised 0‥1
            accumulate=False,
        )

        # 2) MEAN HEADING (cos θ) -------------------------------------------
        heading = torch.atan2(vxvy[b, ..., 1], vxvy[b, ..., 0])  # (A,T)
        imgs[b, 2].index_put_(
            (yp, xp),
            torch.cos(heading[msk]),
            accumulate=False,
        )

    return imgs.clamp_(0, 1)  # (B,3,H,W)

# ------------------------------------------------------------
#   Main blocks
# ------------------------------------------------------------

class RoadCNN(nn.Module):
    """Simple 4‑stage conv tower → 256‑D embedding."""

    def __init__(self, cin: int = 3, c: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, c, 3, 2, 1), nn.ReLU(),          # (B,c,H/2,W/2)
            nn.Conv2d(c, 2*c, 3, 2, 1), nn.ReLU(),          # (B,2c,H/4,W/4)
            nn.Conv2d(2*c, 4*c, 3, 2, 1), nn.ReLU(),        # (B,4c,H/8,W/8)
            nn.Conv2d(4*c, 4*c, 3, 2, 1), nn.ReLU(),        # (B,4c,H/16,W/16)
            nn.AdaptiveMaxPool2d(1),                       # (B,4c,1,1)
        )
        self.out = nn.Linear(4*c, 256)                      # (B,256)

    def forward(self, img: torch.Tensor) -> torch.Tensor:   # img (B,3,H,W)
        feat = self.net(img).flatten(1)                    # (B,4c)
        return self.out(feat)                              # (B,256)


class SocialAttention(nn.Module):
    """Transformer Encoder over 50 agents.  Returns ego token embedding."""

    def __init__(self, dim: int = 128, heads: int = 4, layers: int = 3):
        super().__init__()
        self.input_fc = nn.Linear(10, dim)                 # node‑feat 10 → dim
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.out_fc = nn.Linear(dim, 256)

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        """node_feat (B,50,10) → context (B,256)"""
        x = self.input_fc(node_feat)                       # (B,50,dim)
        h = self.encoder(x)                                # (B,50,dim)
        ego_h = h[:, 0, :]                                 # (B,dim)
        return self.out_fc(ego_h)                          # (B,256)


class CausalBlock(nn.Module):
    """1‑D dilated causal Conv(3) + ReLU."""

    def __init__(self, c_in: int, c_out: int, dilation: int):
        super().__init__()
        pad = (3 - 1) * dilation                           # causal padding
        self.conv = nn.Conv1d(c_in, c_out, 3, dilation=dilation, padding=pad)

    def forward(self, x):                                  # x (B,C,L)
        y = self.conv(x)                                   # (B,c_out,L+pad)
        # Drop look‑ahead frames (first `pad` entries)
        return F.relu(y[..., :-self.conv.dilation[0]])     # (B,c_out,L)


class EgoTCN(nn.Module):
    """3‑layer dilated TCN over 50‑step ego kinematics."""

    def __init__(self, c_in: int = 6):
        super().__init__()
        layers, chans, dil = [], [64, 128, 256], 1
        for c_out in chans:
            layers.append(CausalBlock(c_in, c_out, dil))   # (B,c_out,L)
            c_in = c_out; dil *= 2
        self.net = nn.Sequential(*layers)
        self.glob_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, seq):                                # seq (B,6,50)
        h = self.net(seq)                                  # (B,256,50)
        return self.glob_pool(h).squeeze(-1)               # (B,256)


class EgoLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, seq):               # seq (B, 50, 6)  ← note time-major
        _, (h_n, _) = self.lstm(seq)      # h_n (num_layers, B, 256)
        return h_n[-1]                    # (B, 256)  ← matches old EgoTCN

# ------------------------------------------------------------
#   Full hybrid model
# ------------------------------------------------------------

class HybridTrajNet(nn.Module):
    """Fuse *road pattern*, *social context*, *ego momentum* → 60×2 future."""

    def __init__(self):
        super().__init__()
        self.road   = RoadCNN()
        self.social = SocialAttention()
        self.ego    = EgoLSTM()

        self.fuse = nn.Sequential(
            nn.Linear(256 * 3, 512), nn.ReLU(),
            nn.Linear(512, 512),     nn.ReLU(),
        )                                                               # (B,512)
        self.decoder = nn.Linear(512, 60 * 2)                          # (B,120)

    # ------------------------------------------------------
    def forward(self, data) -> torch.Tensor:
        """Assumes *data.x* is shaped (B × 50, 50, 6) as per existing loader."""
        B = data.num_graphs                                           # batch size

        # (1) reshape historical tensor ------------------------------
        hist = data.x.view(B, 50, 50, 6)                              # (B,50,50,6)

        # (2) optional scale to physical units -----------------------
        if hasattr(data, "scale"):
            scale = data.scale.view(B, 1, 1)                          # (B,1,1)
        else:
            scale = torch.ones(B, 1, 1, device=hist.device)           # (B,1,1)

        # ------- 1) road pattern CNN --------------------------------
        img     = rasterise(hist)                                     # (B,3,H,W)
        road_f  = self.road(img)                                      # (B,256)

        # ------- 2) social attention --------------------------------
        last = hist[:, :, -1]                                         # (B,50,6)
        rel  = last.clone()
        rel[..., :2] -= last[:, 0:1, :2]                              # ego‑centred (B,50,6)

        vxvy = rel[..., 2:4] * scale                                  # (B,50,2)
        axay = rel[..., 4:6] * scale                                  # (B,50,2)
        r    = torch.norm(rel[..., :2] * scale, dim=-1, keepdim=True) # (B,50,1)

        phi  = torch.atan2(rel[..., 1], rel[..., 0])                  # (B,50)
        cos_phi = torch.cos(phi).unsqueeze(-1)                        # (B,50,1)
        sin_phi = torch.sin(phi).unsqueeze(-1)                        # (B,50,1)

        # ego indicator (B,50,1) ------------------------------------
        ego_flag = torch.zeros(B, 50, 1, device=hist.device)
        ego_flag[:, 0, 0] = 1.0

        node = torch.cat(
            [rel[..., :2], vxvy, axay, r, cos_phi, sin_phi, ego_flag], dim=-1
        )                                                             # (B,50,10)
        social_f = self.social(node)                                  # (B,256)

        # ------- 3) ego momentum TCN --------------------------------
        ego_seq = hist[:, 0].permute(0, 1, 2)                         # (B,50,6)
        ego_f   = self.ego(ego_seq)                                   # (B,256)

        # -------- fuse & decode ------------------------------------
        fused = self.fuse(torch.cat([road_f, social_f, ego_f], dim=-1))  # (B,512)
        traj  = self.decoder(fused).view(B, 60, 2)                    # (B,60,2)

        return traj                                                   # still normalised

