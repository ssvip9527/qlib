# 版权所有 (c) 微软公司。
# MIT许可证授权。

from __future__ import annotations

from typing import List, Tuple, cast

import torch
import torch.nn as nn
from tianshou.data import Batch

from qlib.typehint import Literal

from .interpreter import FullHistoryObs

__all__ = ["Recurrent"]


class Recurrent(nn.Module):
    """`OPD <https://seqml.github.io/opd/opd_aaai21_supplement.pdf>`_中提出的网络架构。

    在每个时间步，策略网络的输入分为两部分：
    公共变量和私有变量，分别由本网络中的``raw_rnn``和``pri_rnn``处理。

    一个小的区别是，在此实现中，我们不假设方向是固定的。
    因此添加了另一个``dire_fc``来生成额外的方向相关特征。
    """

    def __init__(
        self,
        obs_space: FullHistoryObs,
        hidden_dim: int = 64,
        output_dim: int = 32,
        rnn_type: Literal["rnn", "lstm", "gru"] = "gru",
        rnn_num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_sources = 3

        rnn_classes = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

        self.rnn_class = rnn_classes[rnn_type]
        self.rnn_layers = rnn_num_layers

        self.raw_rnn = self.rnn_class(hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers)
        self.prev_rnn = self.rnn_class(hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers)
        self.pri_rnn = self.rnn_class(hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers)

        self.raw_fc = nn.Sequential(nn.Linear(obs_space["data_processed"].shape[-1], hidden_dim), nn.ReLU())
        self.pri_fc = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.dire_fc = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self._init_extra_branches()

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_sources, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def _init_extra_branches(self) -> None:
        pass

    def _source_features(self, obs: FullHistoryObs, device: torch.device) -> Tuple[List[torch.Tensor], torch.Tensor]:
        bs, _, data_dim = obs["data_processed"].size()
        data = torch.cat((torch.zeros(bs, 1, data_dim, device=device), obs["data_processed"]), 1)
        cur_step = obs["cur_step"].long()
        cur_tick = obs["cur_tick"].long()
        bs_indices = torch.arange(bs, device=device)

        position = obs["position_history"] / obs["target"].unsqueeze(-1)  # [bs, num_step]
        steps = (
            torch.arange(position.size(-1), device=device).unsqueeze(0).repeat(bs, 1).float()
            / obs["num_step"].unsqueeze(-1).float()
        )  # [bs, num_step]
        priv = torch.stack((position.float(), steps), -1)

        data_in = self.raw_fc(data)
        data_out, _ = self.raw_rnn(data_in)
        # as it is padded with zero in front, this should be last minute
        data_out_slice = data_out[bs_indices, cur_tick]

        priv_in = self.pri_fc(priv)
        priv_out = self.pri_rnn(priv_in)[0]
        priv_out = priv_out[bs_indices, cur_step]

        sources = [data_out_slice, priv_out]

        dir_out = self.dire_fc(torch.stack((obs["acquiring"], 1 - obs["acquiring"]), -1).float())
        sources.append(dir_out)

        return sources, data_out

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        输入应该是一个至少包含以下内容的字典：

        - data_processed: [N, T, C]
        - cur_step: [N]  (int)
        - cur_time: [N]  (int)
        - position_history: [N, S]  (S is number of steps)
        - target: [N]
        - num_step: [N]  (int)
        - acquiring: [N]  (0 or 1)
        """

        inp = cast(FullHistoryObs, batch)
        device = inp["data_processed"].device

        sources, _ = self._source_features(inp, device)
        assert len(sources) == self.num_sources

        out = torch.cat(sources, -1)
        return self.fc(out)


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q_net = nn.Linear(in_dim, out_dim)
        self.k_net = nn.Linear(in_dim, out_dim)
        self.v_net = nn.Linear(in_dim, out_dim)

    def forward(self, Q, K, V):
        q = self.q_net(Q)
        k = self.k_net(K)
        v = self.v_net(V)

        attn = torch.einsum("ijk,ilk->ijl", q, k)
        attn = attn.to(Q.device)
        attn_prob = torch.softmax(attn, dim=-1)

        attn_vec = torch.einsum("ijk,ikl->ijl", attn_prob, v)

        return attn_vec
