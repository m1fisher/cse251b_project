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
D_MODEL       = 128       # embedding size after AgentEncoder
N_AGENTS      = 50       # including the ego at index 0
SEQ_LEN       = 50       # number of past timesteps fed in
HORIZON       = 60       # forecast length (x,y) pairs
GAT_HEADS     = 32        # multi‑head attention
LSTM_HIDDEN   = 512      # hidden units in temporal LSTM
LSTM_LAYERS   = 3
POOL_HEADS = 7

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
        # TODO: Consider graph 1 x 50, ego to all other agents
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


def build_ego_star_edges(
        n_agents: int,
        seq_len : int,
        ego_idx : int = 0,
        bidirectional: bool = False,
        device=None
    ) -> torch.Tensor:
    """
    Returns edge_index of shape (2, E) where
      •  source  = any non-ego node (a ≠ ego_idx, any t_src)
      •  target  = the ego node             (ego_idx, any t_dst)
    If `bidirectional` is True we also add the reverse edge so the ego can
    send messages back to the neighbours.

    Nodes are enumerated as  node_id = t * n_agents + a .
    """
    A, T = n_agents, seq_len
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # All node indices except the ego’s within ONE frame
    non_ego = torch.arange(A, device=device)
    non_ego = non_ego[non_ego != ego_idx]                 # (A-1,)

    # Pre-allocate lists to collect edges
    src_list, dst_list = [], []

    for t_dst in range(T):
        ego_dst = t_dst * A + ego_idx                     # scalar

        # ----- edges from *all* timesteps to this ego node -----
        for t_src in range(T):
            offset = t_src * A
            src_nodes = non_ego + offset                  # (A-1,)
            dst_nodes = torch.full_like(src_nodes, ego_dst)
            src_list.append(src_nodes)
            dst_list.append(dst_nodes)

            if bidirectional:
                # reverse: ego → neighbours (same pair count)
                src_list.append(dst_nodes)                # ego as src
                dst_list.append(src_nodes)                # neighbours dst

    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    edge_index = torch.stack([src, dst], dim=0)           # (2, E)
    return edge_index


class EgoStarSTGAT(nn.Module):
    def __init__(self, n_agents=50, seq_len=50,
                 hidden=128, heads=GAT_HEADS, ego_idx=0, **kw):
        super().__init__()
        self.A, self.T, self.ego_idx = n_agents, seq_len, ego_idx
        self.gat = GATConv(        # or GATConv
            in_channels  = hidden,
            out_channels = 4 * hidden // heads,
            heads        = heads,
            concat       = True,     # or True, your choice
            add_self_loops=False
        )
        edge = build_ego_star_edges(n_agents, seq_len, ego_idx)
        self.register_buffer("edge_index_base", edge)     # (2, E)

    def forward(self, h):              # h : (B, A, T, D)
        B, A, T, D = h.shape
        # (B, T, A, D) → (B, N, D) with N = A·T
        h = h.permute(0, 2, 1, 3).reshape(B, T * A, D)
        h = h.reshape(-1, D)           # flatten batch dimension

        # replicate edge_index for every batch
        N = T * A
        offset = (torch.arange(B, device=h.device) * N
                  ).repeat_interleave(self.edge_index_base.size(1))
        edge_index = self.edge_index_base.repeat(1, B) + offset

        out = self.gat(h, edge_index)  # (B·N, D)
        out = out.reshape(B, T, A, D).permute(0, 2, 1, 3)
        return out


class MultiHeadPool(nn.Module):
    def __init__(self, n_heads=POOL_HEADS, d_model=D_MODEL):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_heads, d_model))
        self.scale = d_model ** -0.5
    def forward(self, ego, others):
        # ego: (B, T, D_MODEL), others: (B, A-1, T, D_MODEL)
        B, T, D = ego.shape
        # reshape others to (B, T, N, D)
        others_t = others.permute(0,2,1,3)           # (B, T, N, D)
        # prepare queries → (1,1,H,D)
        q = self.queries.view(1,1, -1, D)
        # keys → (B,T,1,N,D)
        k = others_t.unsqueeze(2)
        # attention logits → (B,T,H,N)
        logits = (k * q.unsqueeze(3)).sum(-1) * self.scale
        weights = torch.softmax(logits, dim=-1)      # (B,T,H,N)
        # weighted sum → (B,T,H,D)
        ctx = torch.einsum('bthn,btnm->bthm', weights, others_t)
        return ctx  # (B, T, POOL_HEADS, D_MODEL)


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

class GlobalLSTM(nn.Module):
    def __init__(self, in_dim=D_MODEL * N_AGENTS, lstm_hidden=LSTM_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, lstm_hidden, num_layers=LSTM_LAYERS,
                             batch_first=True)
        self.head = nn.Linear(lstm_hidden, HORIZON * 2)

    def forward(self, all_agents_seq):
        """all_agents_seq: (B, SEQ_LEN, N_AGENTS * D_MODEL)"""
        def lstm_forward(seq):
            out, _ = self.lstm(seq)
            return out
        out = cp.checkpoint(lstm_forward, all_agents_seq, use_reentrant=False)  # heavy part we checkpoint
        last = out[:, -1, :]                          # last hidden
        pred = self.head(last)                        # (B, 120)
        return pred.view(-1, HORIZON, 2)              # (B, 60, 2)


class PooledLSTM(nn.Module):
    def __init__(self, in_dim=(1+POOL_HEADS)*D_MODEL, hidden=LSTM_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=LSTM_LAYERS,
                             batch_first=True)
        self.head = nn.Linear(hidden, HORIZON*2)
    def forward(self, seq):
        # seq: (B, T, in_dim)
        def run_lstm(s):
            out, _ = self.lstm(s)
            return out
        # checkpoint LSTM to save memory
        out = cp.checkpoint(run_lstm, seq, use_reentrant=False)
        last = out[:, -1, :]
        return self.head(last).view(-1, HORIZON, 2)


# ------------------------------------------------------------
# 4) Full model wrapper
# ------------------------------------------------------------
class SocialLSTMPredictor(nn.Module):
    """AgentEncoder ➜ Social GAT ➜ Ego LSTM ➜ 60‑step (x,y) forecast"""
    def __init__(self):
        super().__init__()
        # TODO: Make sure this encoder layer is useful; could try without it
        self.encoder = AgentEncoder()
        # self.social = nn.Sequential(
        #     SocialGAT(),  # first GAT layer
        #     nn.ReLU(),  # non‑linearity between GATs
        #     SocialGAT()  # second GAT layer
        # )
        self.social = EgoStarSTGAT()
        self.pool = MultiHeadPool()
        self.temporal = PooledLSTM()

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
        # TODO: Check dimensionality and make sure that all necessary information
        # is getting passed here.
        # TRY passing all of h?
        ego = h[:, 0]  # (B,T,D)
        others = h[:, 1:]  # (B,49,T,D)
        ctx = self.pool(ego, others)  # (B,T,POOL_HEADS,D)
        ctx_flat = ctx.view(B, SEQ_LEN, POOL_HEADS * D_MODEL)
        seq = torch.cat([ego, ctx_flat], dim=-1)  # (B,T,(1+H)*D)
        # all_seq = h.permute(0, 2, 1, 3).contiguous().view(B, SEQ_LEN, -1)
        return self.temporal(seq)                  # (B, 60, 2)

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
