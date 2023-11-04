import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=1000):
        """
        Create sinusoidal timestep embeddings.
        :param t: (N) Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, max_period=self.num_timesteps)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, cond_seq_len, cond_size, dropout_prob, cond_hidden_size=256):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        # TODO try LSTM
        self.mlp = nn.Sequential(
            nn.Linear(cond_size, cond_hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(cond_hidden_size, cond_hidden_size, bias=True),
        )
        self.dropout_prob = dropout_prob

    def token_drop(self, cond, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        for i in range(cond.shape[0]):
            if drop_ids[i]:
                cond[i] = torch.zeros((cond.shape[1], cond.shape[2]), device=cond.device)
        return cond

    def forward(self, cond, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        embeddings = self.mlp(cond)
        return embeddings