"""
Social‑Aware Trajectory Predictor
================================
AgentEncoder ➜ Social GAT ➜ Temporal LSTM ➜ Forecast Head
---------------------------------------------------------
* 50 agents, 50 past steps, 6‑dim features  →  60‑step (x,y) forecast for ego
* Torch + Torch Geometric (GATConv)
* Written in the clear‑header / inline‑comment style the user prefers.

Author: ChatGPT (o3)
"""

# ------------------------------------------------------------
# Imports & configuration
# ------------------------------------------------------------
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.utils.checkpoint as cp

# Model hyper‑parameters (feel free to tweak)
D_IN          = 6        # x, y, vx, vy, heading, type
D_MODEL       = 128      # embedding size after AgentEncoder
N_AGENTS      = 50       # including the ego at index 0
SEQ_LEN       = 50       # number of past timesteps fed in
HORIZON       = 60       # forecast length (x,y) pairs
GAT_HEADS     = 8        # multi‑head attention
LSTM_HIDDEN   = 1024     # hidden units in temporal LSTM
LSTM_LAYERS   = 3

# ------------------------------------------------------------
# 1) AgentEncoder – small shared MLP (6 ➜ 128)
# ------------------------------------------------------------
class AgentEncoder(nn.Module):
    def __init__(self, in_dim=D_IN, hidden=D_MODEL):
        super().__init__()
        ## layer norm added to reduce dead ReLU problem
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )

    def forward(self, x):
        """x shape: (B, N_AGENTS, SEQ_LEN, 6)"""
        B, A, T, F = x.shape
        x = x.reshape(B * A * T, F)
        z = self.net(x)                                   # (B*A*T, D_MODEL)
        return z.reshape(B, A, T, D_MODEL)               # (B, A, T, D_MODEL)

# ------------------------------------------------------------
# 2) Social‑Attention – a single GATConv applied *per timestep*
# ------------------------------------------------------------
class SocialGAT(nn.Module):
    def __init__(self, hidden=D_MODEL, heads=GAT_HEADS):
        super().__init__()
        self.gat = GATConv(hidden, hidden // heads, heads=heads, add_self_loops=False)
        # Pre‑compute fully‑connected edge_index once (minus self‑loops)
        src, dst = torch.meshgrid(torch.arange(N_AGENTS), torch.arange(N_AGENTS), indexing="ij")
        mask = src != dst
        self.register_buffer("edge_index", torch.stack([src[mask], dst[mask]], dim=0))  # (2, A*(A-1))

    def forward(self, h):
        """h shape: (B, A, T, D_MODEL)"""
        B, A, T, D = h.shape
        h = h.permute(0, 2, 1, 3).contiguous()           # (B, T, A, D)
        h = h.reshape(B * T, A, D)                       # merge batch & time

        # GATConv expects (num_nodes, feat) so flatten agents first
        h = h.reshape(-1, D)                             # (B*T*A, D)
        edge_index = self.edge_index                     # fixed on CUDA/CPU
        # Repeat edge_index for each graph in the batch (B*T graphs)
        offset = torch.arange(0, A * B * T, A, device=h.device)
        offset = offset.repeat_interleave(edge_index.size(1))
        edge_index_bt = edge_index.repeat(1, B * T) + offset

        h_out = self.gat(h, edge_index_bt)               # (B*T*A, D)
        return h_out.reshape(B, T, A, D).permute(0, 2, 1, 3)  # (B, A, T, D)

# ------------------------------------------------------------
# 3) Temporal model – LSTM over the *ego* embeddings (T, D) ➜ (H)
# ------------------------------------------------------------
class EgoLSTM(nn.Module):
    def __init__(self, hidden=D_MODEL, lstm_hidden=LSTM_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(hidden, lstm_hidden, num_layers=LSTM_LAYERS, batch_first=True)
        self.head = nn.Linear(lstm_hidden, HORIZON * 2)

    def forward(self, ego_seq):
        """
        ego_seq: (B, SEQ_LEN, D_MODEL)
        Using Checkpoint to prevent saving of some intermediate activations -> reduced runtime but reduced VRAM overhead
        """
        # ego_seq : (B, 50, D)
        def lstm_forward(seq):
            out, _ = self.lstm(seq)
            return out
        # 1) run forward with checkpointing
        out = cp.checkpoint(lstm_forward, ego_seq, use_reentrant=False)  # heavy part we checkpoint
        # 2) same head as before
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.view(-1, HORIZON, 2)                # (B, 60, 2)

# ------------------------------------------------------------
# 4) Full model wrapper
# ------------------------------------------------------------
class SocialLSTMPredictor(nn.Module):
    """AgentEncoder ➜ Social GAT ➜ Ego LSTM ➜ 60‑step (x,y) forecast"""
    def __init__(self):
        super().__init__()
        self.encoder = AgentEncoder()
        self.social = nn.Sequential(
            SocialGAT(),  # first GAT layer
            nn.ReLU(),  # non‑linearity between GATs
            SocialGAT()  # second GAT layer
        )
        self.temporal = EgoLSTM()

    def forward(self, data):
        """
        Accepts either a raw tensor **or** a PyG `Batch`/`Data` object.

        Parameters
        ----------
        data :
            • `Tensor (B, N_AGENTS, SEQ_LEN, 6)` – already reshaped
              **or**
            • PyG `Data`/`Batch` whose `.x` is flattened to
              `(B * N_AGENTS, SEQ_LEN, 6)` and has `num_graphs`.

        Returns
        -------
        Tensor (B, 60, 2) – future (x,y) for the ego.
        """
        # ----------------------------------------------------
        # 0) Prepare x so that shape = (B, A, T, 6)
        # ----------------------------------------------------
        if hasattr(data, "x") and hasattr(data, "num_graphs"):
            B = data.num_graphs                         # scenes in batch
            x = data.x.view(B, N_AGENTS, SEQ_LEN, D_IN)
        else:
            # Assume caller already passed a tensor in the right shape
            x = data
            B = x.size(0)

        # 1) Encode each agent/time‑step feature vector
        h = self.encoder(x)                             # (B, A, T, D_MODEL)

        # 2) Social attention per time‑step
        h = self.social(h)                              # same shape

        # 3) Extract ego (agent 0) sequence and feed to temporal LSTM
        ego_seq = h[:, 0]                               # (B, T, D_MODEL)
        return self.temporal(ego_seq)                   # (B, 60, 2)

# ------------------------------------------------------------
# 5) Quick unit test + parameter count
# ------------------------------------------------------------
if __name__ == "__main__":
    BATCH = 4
    dummy = torch.randn(BATCH, N_AGENTS, SEQ_LEN, D_IN)  # fake data
    model = SocialLSTMPredictor()

    # Forward pass sanity‑check
    out = model(dummy)
    print("Output shape:", out.shape)                     # Expected: (4, 60, 2)

    # Parameter count breakdown
    print("\nParameter breakdown:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total_params += count
            print(f"{name:40s} : {count:10,}")
    print(f"\nTotal trainable parameters: {total_params:,}")
