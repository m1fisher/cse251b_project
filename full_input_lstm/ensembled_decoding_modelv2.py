# models.py — 512×512 raster, **cross‑fused ego/road sequence** all 1024‑D
# =====================================================================
# Major upgrade:
#   • RoadCNN now returns *both* a spatial feature map (B,256,16,16)
#     *and* a global 1024‑D road vector used in the fusion head.
#   • Ego history at every time step is fused with the **bilinearly‑sampled
#     road feature under the ego footprint** before feeding a 2‑layer LSTM.
#   • All three branches (road_global, social, ego_f) emit 1024‑D vectors.
#   • Decoder unchanged: Linear(1024 → 120) → reshape (B,60,2).
#   • Fully shape‑annotated.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- Hyper‑params -----------------------------
CELL_RES   = 0.5        # metres per input pixel
RASTER_SZ  = 512        # 512 × 512 raster (≈ 256 m window)
STRIDES    = 4          # four stride‑2 convs ⇒ 2**4 reduction
FEAT_C     = 256        # channels in road feature map
BRANCH_DIM = 1024       # per‑branch output size

# ----------------------------------------------------------------------
# Helper: rasterise trajectories
# ----------------------------------------------------------------------

def rasterise(hist, grid_size: int = RASTER_SZ, cell_res: float = CELL_RES):
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

# ----------------------------------------------------------------------
# Road CNN — returns *feature map* + global vector
# ----------------------------------------------------------------------

class RoadCNN(nn.Module):
    def __init__(self, cin: int = 3, c: int = 32):
        super().__init__()
        layers = []
        in_c = cin
        for _ in range(STRIDES):           # 4 stride‑2 convs
            layers += [nn.Conv2d(in_c, c, 3, 2, 1), nn.ReLU()]
            in_c = c
            c *= 2
        self.conv_stack = nn.Sequential(*layers)          # (B,FEAT_C,16,16)
        self.glob_pool  = nn.AdaptiveMaxPool2d(1)
        self.head_fc    = nn.Linear(FEAT_C, BRANCH_DIM)

    # ------------------------------------------------------------------
    def forward(self, img):
        fmap = self.conv_stack(img)                       # (B,256,16,16)
        glob = self.glob_pool(fmap).flatten(1)            # (B,256)
        road_vec = self.head_fc(glob)                     # (B,1024)
        return road_vec, fmap

# ----------------------------------------------------------------------
# Social‑attention module → 1024‑D
# ----------------------------------------------------------------------

class SocialAttention(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4, layers: int = 3):
        super().__init__()
        self.in_fc = nn.Linear(10, dim)
        enc_layer  = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*2, batch_first=True, activation="gelu")
        self.encoder= nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_fc = nn.Linear(dim, BRANCH_DIM)

    def forward(self, node_feat):           # (B,50,10)
        x = self.in_fc(node_feat)           # (B,50,dim)
        h = self.encoder(x)
        ego_h = h[:, 0, :]                  # (B,dim)
        return self.out_fc(ego_h)           # (B,1024)

# ----------------------------------------------------------------------
# Ego LSTM with road‑feature conditioning at each step
# ----------------------------------------------------------------------

class EgoCondLSTM(nn.Module):
    def __init__(self, road_channels: int = FEAT_C):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6 + road_channels, hidden_size=BRANCH_DIM, num_layers=2, batch_first=True)

    def forward(self, ego_seq, road_seq):   # ego_seq (B,T,6); road_seq (B,T,256)
        x = torch.cat([ego_seq, road_seq], dim=-1)        # (B,T,262)
        _, (h, _) = self.lstm(x)
        return h[-1]                                      # (B,1024)

# ----------------------------------------------------------------------
# HybridTrajNet (road ⊕ social ⊕ ego) → future 60×2
# ----------------------------------------------------------------------

class HybridTrajNetv2(nn.Module):
    """Fuse local‑road, social, and conditioned‑ego branches."""

    def __init__(self):
        super().__init__()
        self.road   = RoadCNN()
        self.social = SocialAttention()
        self.ego    = EgoCondLSTM()

        self.fuse = nn.Sequential(
            nn.Linear(BRANCH_DIM * 3, BRANCH_DIM), nn.ReLU(),
            nn.Linear(BRANCH_DIM, BRANCH_DIM),     nn.ReLU(),
        )
        self.decoder = nn.Linear(BRANCH_DIM, 60 * 2)

    # ---------------------------------------------------------
    def forward(self, data):
        B = data.num_graphs
        hist = data.x.view(B, 50, 50, 6)                  # (B,A=50,T=50,6)

        # -------- 1) Raster + Road CNN --------
        img = rasterise(hist)                             # (B,3,512,512)
        road_vec, fmap = self.road(img)                   # (B,1024) & (B,256,16,16)

        # -------- 2) Social attention ---------
        last = hist[:, :, -1]                             # (B,50,6)
        rel  = last.clone()
        rel[..., :2] -= last[:, 0:1, :2]                  # Δx,Δy to ego
        vxvy = rel[..., 2:4]
        axay = rel[..., 4:6]
        r    = torch.norm(rel[..., :2], dim=-1, keepdim=True)
        phi  = torch.atan2(rel[...,1], rel[...,0])
        ego_flag = torch.zeros(B, 50, 1, device=hist.device)
        ego_flag[:, 0, 0] = 1.0
        node = torch.cat([
            rel[..., :2], vxvy, axay, r,
            torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1), ego_flag
        ], dim=-1)                                       # (B,50,10)
        social_vec = self.social(node)                   # (B,1024)

        # -------- 3) Sample road features along ego path --------
        C, H, W = fmap.shape[1:]                         # 256,16,16
        T = hist.shape[2]
        # ego (x,y) in metres for each t
        ego_xy = hist[:, 0, :, :2]                       # (B,T,2)

        cell_size = CELL_RES * (2 ** STRIDES)            # ≈ 0.5 × 16 = 8 m per feature‑map px
        # normalise to [-1,1] for grid_sample
        x_norm = 2 * (ego_xy[..., 0] / (cell_size * (W - 1)))
        y_norm = 2 * (-ego_xy[..., 1] / (cell_size * (H - 1)))
        grid = torch.stack([x_norm, -y_norm], dim=-1)  # (B, T, 2)
        grid = grid.unsqueeze(2)  # (B, T, 1, 2)

        # differentiable lookup
        road_seq = (
            F.grid_sample(fmap, grid, align_corners=True)  # (B, C, T, 1)
            .squeeze(-1)  # (B, C, T)
            .transpose(1, 2)  # (B, T, C=256 or 1024)
        )

        # grid = torch.stack([x_norm, y_norm], dim=-1).view(B, T, 1, 1, 2)
        # road_seq = F.grid_sample(fmap, grid, align_corners=True).squeeze(-1).squeeze(-1).transpose(1, 2)  # (B,T,256)
        #
        # -------- 4) Ego‑conditioned LSTM --------
        ego_seq = hist[:, 0]                              # (B,T,6)
        ego_vec = self.ego(ego_seq, road_seq)            # (B,1024)

        # -------- 5) Fuse three branches --------
        fused = self.fuse(torch.cat([road_vec, social_vec, ego_vec], dim=-1))  # (B,1024)
        traj  = self.decoder(fused).view(B, 60, 2)
        return traj
