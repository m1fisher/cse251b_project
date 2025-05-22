import torch.nn as nn

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

