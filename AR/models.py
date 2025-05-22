import torch.nn as nn
import torch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=1024, output_dim=60 * 2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        x= x.reshape(-1, 50, 50, 6)  # (batch_size, num_agents, seq_len, input_dim)
        x = x[:, 0, :, :] # Only Consider ego agent index 0

        lstm_out, _ = self.lstm(x)
        # lstm_out is of shape (batch_size, seq_len, hidden_dim) and we want the last time step output
        out = self.fc(lstm_out[:, -1, :])
        return out.view(-1, 60, 2)

class LinearForecast(nn.Module):
    """ Baseline linear regression (GPT4o GENERATED) """
    def __init__(self, input_dim=6, seq_len=50, output_steps=60):
        super().__init__()
        self.seq_len    = seq_len
        self.input_dim  = input_dim
        self.output_dim = output_steps * 2    # 2 coords per step

        # one linear layer from 50*6 → 60*2
        self.fc = nn.Linear(self.seq_len * self.input_dim,
                            self.output_dim)

    def forward(self, data):
        # data.x: assumed flat batch e.g. [B*agents,50,6] or raw [B,agents,50,6]
        x = data.x
        # match your LSTM reshape:
        x = x.reshape(-1, 50, 50, 6)   # (B, agents, seq_len, input_dim)
        ego = x[:, 0, :, :]           # pick agent 0 → (B, seq_len, input_dim)

        # flatten time & features → (B, 50*6)
        B = ego.size(0)
        feat = ego.view(B, -1)

        # linear regression → (B, 120)
        out = self.fc(feat)

        # reshape to (B, 60, 2)
        return out.view(B, -1, 2)


########################################NEW#####################################

# -----------------------------------------------------------------------------#
#  helper: causal 1-D convolution (no future leakage)                          #
# -----------------------------------------------------------------------------#
class CausalConv1d(nn.Module):
    """Conv1d with causal padding – output length == input length."""
    def __init__(self, cin, cout, k=3, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k,
                              padding=self.pad,
                              dilation=dilation)

    def forward(self, x):                                # x: (N, C_in, L)
        y = self.conv(x)
        if self.pad:                                     # strip future steps
            y = y[..., :-self.pad]
        return y

# -----------------------------------------------------------------------------#
#  Temporal block: dilated causal TCN with residual                             #
# -----------------------------------------------------------------------------#
class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(cin, cout, k, dilation)
        self.conv2 = CausalConv1d(cout, cout, k, dilation)
        self.down  = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.conv.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.conv.weight, nonlinearity='relu')

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out + self.down(x)                        # residual

class TemporalConvNet(nn.Module):
    def __init__(self, c_in, hidden=64):
        super().__init__()
        layers = []
        for d in (1, 2, 4):                              # dilations 1–2–4
            layers.append(TemporalBlock(c_in, hidden, k=3, dilation=d))
            c_in = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):                                # x: (N, C, L)
        return self.net(x)

# -----------------------------------------------------------------------------#
#  ST-GNN                                                                        #
# -----------------------------------------------------------------------------#
class STGNNOneStep(nn.Module):
    """
    Predict the next frame (6-D) for all 50 agents given 50-frame history.

    Inputs:  batch.x   – shape (B*50, 50, 6)
             batch.batch – graph id (unused except for B derivation)

    Returns: tensor (B, 50, 6)
    """
    def __init__(self,
                 in_dim:   int = 6,
                 hidden:   int = 64,
                 num_agents: int = 50,
                 hist_len: int = 50):
        super().__init__()
        self.A = num_agents
        self.T = hist_len
        # --- fully-connected edge index for 50-node graph -------------------
        src, dst = torch.meshgrid(torch.arange(self.A),
                                  torch.arange(self.A), indexing='ij')
        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # drop self-loops
        self.register_buffer('edge_index_base', edge_index, persistent=False)

        # --- spatial encoder (shared across frames) -------------------------
        self.gat1 = GATv2Conv(in_dim,  hidden // 4, heads=4, add_self_loops=False)
        self.gat2 = GATv2Conv(hidden,  hidden // 4, heads=4, add_self_loops=False)

        # --- temporal causal TCN (shared across agents) ---------------------
        self.tcn  = TemporalConvNet(hidden, hidden)

        # --- decoder --------------------------------------------------------
        self.head = nn.Sequential(
            nn.Conv1d(hidden, 32, 1),
            nn.SiLU(),
            nn.Conv1d(32, 6, 1)
        )

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # --------------------------------------------------------------------- #
    def repeat_edge_index(self, B: int, device: torch.device):
        """
        Replicate the 50-node fully-connected edge_index for every graph
        in the batch with correct index offsets: [2, E*B].
        """
        ei = self.edge_index_base
        offset = (torch.arange(B, device=device) * self.A).repeat_interleave(ei.size(1))
        return ei.repeat(1, B) + offset

    # --------------------------------------------------------------------- #
    def forward(self, batch):
        x      = batch.x                               # (B*50, 50, 6)
        B      = batch.num_graphs
        device = x.device

        # Reshape into (B, A, T, F)
        x = x.view(B, self.A, self.T, -1)
        # ------------------------------------------------------------------ #
        #  Spatial pass – per timestep graph attention                       #
        # ------------------------------------------------------------------ #
        edge_index_batch = self.repeat_edge_index(B, device)
        spatial_out = []
        for t in range(self.T):
            feats_t = x[:, :, t, :]                    # (B, A, F)
            feats_t = feats_t.reshape(B * self.A, -1)  # flatten graphs
            h = F.relu(self.gat1(feats_t, edge_index_batch))
            h = F.relu(self.gat2(h,       edge_index_batch))
            spatial_out.append(h.view(B, self.A, -1))
        h = torch.stack(spatial_out, dim=3)            # (B, A, H, T)

        # ------------------------------------------------------------------ #
        #  Temporal pass – causal TCN per agent                              #
        # ------------------------------------------------------------------ #
        h = h.reshape(B * self.A, -1, self.T)          # (B*A, H, T)
        h = self.tcn(h)                                # (B*A, H, T)
        h = h[..., -1]                                 # pick last timestep  (B*A, H)

        # ------------------------------------------------------------------ #
        #  Decode to 6-D next-frame prediction                               #
        # ------------------------------------------------------------------ #
        y = self.head[0](h.unsqueeze(-1))              # 1×1 conv keeps shape
        y = self.head[1](y)
        y = self.head[2](y).squeeze(-1)                # (B*A, 6)
        y = y.view(B, self.A, 6)                       # (B, 50, 6)
        return y
