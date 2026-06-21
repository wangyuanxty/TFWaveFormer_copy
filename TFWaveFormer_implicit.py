"""
TFWaveFormer with Implicit Neural Wavelet Scales

K learnable continuous kernel sizes, kernel weights queried from an implicit MLP.
Per-channel kernel customization: MLP(scale, position, channel_id) → weight.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math
from models.modules import TimeEncoder
from utils.utils import NeighborSampler


# ════════════════════════════════════════ wavelet filter ════════════════════════════════════════

class ImplicitWaveletFilter(nn.Module):
    def __init__(self, d_model: int, K: int = 4, device: str = 'cpu'):
        super().__init__()
        self.d_model, self.K, self.device = d_model, K, device
        self.scale_weights = nn.Parameter(torch.ones(K, d_model) * (1.0 / K))
        self.scale_params = nn.Parameter(torch.rand(K) * 12 + 3)
        self.implicit_net = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x: torch.Tensor, training: bool = True):
        batch_size, d_model, seq_len = x.shape
        outputs = []
        for k in range(self.K):
            s = self.scale_params[k].clamp(3, 15)
            ks = int(s.round().item()); ks = ks + 1 if ks % 2 == 0 else ks; ks = min(ks, 15)
            pad = ks // 2

            # Batch all d_model × ks queries into one MLP call
            pos = torch.linspace(-1, 1, ks, device=self.device)               # (ks,)
            sv = torch.full((ks,), s.item() / 15, device=self.device)          # (ks,)
            ch = torch.arange(d_model, dtype=torch.float, device=self.device) / d_model  # (d_model,)

            # Meshgrid: (d_model, ks, 3)
            pos_g, sv_g, ch_g = torch.meshgrid(pos, sv, ch, indexing='ij')  # wrong order
            # Build (d_model, ks, 3) manually to get [pos, scale, channel] right
            # We want queries of shape (d_model * ks, 3): [pos, scale_val, channel_id]
            pos_r = pos.unsqueeze(0).expand(d_model, ks)           # (d_model, ks)
            sv_r  = sv.unsqueeze(0).expand(d_model, ks)            # (d_model, ks)
            ch_r  = ch.unsqueeze(1).expand(d_model, ks)            # (d_model, ks)
            inp = torch.stack([pos_r, sv_r, ch_r], dim=-1)         # (d_model, ks, 3)
            inp = inp.reshape(d_model * ks, 3)                     # (d_model * ks, 3)

            w = self.implicit_net(inp).view(d_model, ks)    # (d_model, ks)
            w = F.softmax(w, dim=-1)                       # softmax per channel

            # Apply per-channel conv using grouped conv1d
            x_pad = F.pad(x, (pad, pad), mode='replicate')         # (batch, d_model, seq_len + 2*pad)
            w_batch = w.unsqueeze(1)                                # (d_model, 1, ks)
            out = F.conv1d(x_pad, w_batch, groups=d_model)         # (batch, d_model, seq_len)
            outputs.append(out)

        multi_scale = torch.stack(outputs, dim=0)
        sw = torch.softmax(self.scale_weights, dim=0).view(self.K, 1, self.d_model, 1)
        return (multi_scale * sw).sum(dim=0)

    def get_learned_scales(self):
        return self.scale_params.detach().cpu().numpy()

    def get_scale_statistics(self):
        s = self.get_learned_scales()
        return {'mean': float(np.mean(s)), 'std': float(np.std(s)), 'min': float(np.min(s)),
                'max': float(np.max(s)), 'values': s.tolist()}

    def diversity_loss(self):
        return torch.tensor(0.0, device=self.device)

    def anneal_temperature(self, epoch, total_epochs):
        pass


# ════════════════════════════════════════ transformer layer ════════════════════════════════════════

class AdaptiveWaveletTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, max_seq_len, dropout=0.1, device='cpu'):
        super().__init__()
        self.wavelet_filter = ImplicitWaveletFilter(d_model, device=device)
        self.feature_transform = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gate = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, d_model), nn.Sigmoid())
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=True, src_mask=None, src_key_padding_mask=None):
        xw = self.wavelet_filter(x.transpose(1, 2)).transpose(1, 2)
        xw = self.feature_transform(xw)
        xw = xw * self.gate(torch.mean(x, dim=1, keepdim=True))
        x = self.norm1(x + xw)
        a, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.norm2(x + self.dropout(a))
        return self.norm3(x + self.ffn(x))

    def get_scale_statistics(self):
        return self.wavelet_filter.get_scale_statistics()


# ════════════════════════════════════════ helpers ════════════════════════════════════════

class MultiModalAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.query_net = nn.Linear(d_model, d_model)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.output_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        q = self.query_net(torch.mean(x, dim=1, keepdim=True))
        return self.output_proj(self.multihead_attn(q, x, x)[0].squeeze(1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device='cpu'):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1).to(device))

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class NIFEncoder(nn.Module):
    def __init__(self, nif_feat_dim, device='cpu'):
        super().__init__()
        self.nif_feat_dim, self.device = nif_feat_dim, device
        self.nif_encode_layer = nn.Sequential(nn.Linear(1, nif_feat_dim), nn.ReLU(), nn.Linear(nif_feat_dim, nif_feat_dim))

    def count_nodes_appearances(self, src_ids, dst_ids, src_nbrs, dst_nbrs):
        src_app, dst_app = [], []
        for i in range(len(src_ids)):
            su, si, sc = np.unique(src_nbrs[i], return_inverse=True, return_counts=True)
            du, di, dc = np.unique(dst_nbrs[i], return_inverse=True, return_counts=True)
            sm, dm = dict(zip(su, sc)), dict(zip(du, dc))
            if src_ids[i] in dm: sm[src_ids[i]] = dm[src_ids[i]]; dm[src_ids[i]] = dm[src_ids[i]]
            if dst_ids[i] in sm: sm[dst_ids[i]] = sm[dst_ids[i]]; dm[dst_ids[i]] = sm[dst_ids[i]]
            src_app.append(torch.stack([torch.from_numpy(sc[si]).float().to(self.device),
                          torch.tensor([dm.get(n, 0) for n in src_nbrs[i]]).float().to(self.device)], dim=1))
            dst_app.append(torch.stack([torch.tensor([sm.get(n, 0) for n in dst_nbrs[i]]).float().to(self.device),
                          torch.from_numpy(dc[di]).float().to(self.device)], dim=1))
        return torch.stack(src_app, dim=0), torch.stack(dst_app, dim=0)

    def forward(self, src_ids, dst_ids, src_nbrs, dst_nbrs):
        sa, da = self.count_nodes_appearances(src_ids, dst_ids, src_nbrs, dst_nbrs)
        return self.nif_encode_layer(sa.unsqueeze(-1)).sum(dim=2), self.nif_encode_layer(da.unsqueeze(-1)).sum(dim=2)


# ════════════════════════════════════════ main model ════════════════════════════════════════

class TFWaveFormerImplicit(nn.Module):
    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
                 time_feat_dim, channel_embedding_dim, num_layers=2, num_heads=8,
                 dropout=0.1, max_input_sequence_length=128, device='cpu'):
        super().__init__()
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler, self.device = neighbor_sampler, device
        self.node_feat_dim, self.edge_feat_dim = self.node_raw_features.shape[1], self.edge_raw_features.shape[1]
        self.time_feat_dim, self.num_heads, self.dropout = time_feat_dim, num_heads, dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.num_layers = num_layers

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.nif_encoder = NIFEncoder(nif_feat_dim=channel_embedding_dim, device=device)
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(self.node_feat_dim, self.edge_feat_dim),
            'edge': nn.Linear(self.edge_feat_dim, self.edge_feat_dim),
            'time': nn.Linear(time_feat_dim, self.edge_feat_dim),
            'nif': nn.Linear(channel_embedding_dim, self.edge_feat_dim)
        })
        self.reduce_layer = nn.Linear(4 * self.edge_feat_dim, self.edge_feat_dim)
        self.wavelet_transformers = nn.ModuleList([
            AdaptiveWaveletTransformerLayer(self.edge_feat_dim, num_heads, max_input_sequence_length, dropout, device)
            for _ in range(num_layers)
        ])
        self.pos_encoder = PositionalEncoding(self.edge_feat_dim, max_input_sequence_length, device)
        self.final_attention = MultiModalAttention(self.edge_feat_dim, num_heads, dropout)
        self.output_norm = nn.LayerNorm(self.edge_feat_dim)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times):
        sn, se, st = self.neighbor_sampler.get_historical_neighbors(
            node_ids=src_node_ids, node_interact_times=node_interact_times, num_neighbors=self.max_input_sequence_length)
        dn, de, dt = self.neighbor_sampler.get_historical_neighbors(
            node_ids=dst_node_ids, node_interact_times=node_interact_times, num_neighbors=self.max_input_sequence_length)
        snf, dnf = self.nif_encoder(src_node_ids, dst_node_ids, sn, dn)
        sf = self._get_mm_feat(node_interact_times, sn, se, st, snf)
        df = self._get_mm_feat(node_interact_times, dn, de, dt, dnf)
        sf, df = self.pos_encoder(sf), self.pos_encoder(df)
        for wt in self.wavelet_transformers:
            sf, df = wt(sf, training=self.training), wt(df, training=self.training)
        return self.output_norm(self.final_attention(sf)), self.output_norm(self.final_attention(df))

    def _get_mm_feat(self, times, nbrs, edges, nbr_times, nif_feat):
        nf, ef, tf = self.get_features(times, nbrs, edges, nbr_times, self.time_encoder)
        return self.reduce_layer(torch.cat([
            self.projection_layer['node'](nf), self.projection_layer['edge'](ef),
            self.projection_layer['time'](tf), self.projection_layer['nif'](nif_feat)], dim=-1))

    def get_features(self, times, nbrs, edges, nbr_times, time_enc):
        nf = self.node_raw_features[torch.from_numpy(nbrs)]
        ef = self.edge_raw_features[torch.from_numpy(edges)]
        tf = time_enc(timestamps=torch.from_numpy(times[:, np.newaxis] - nbr_times).float().to(self.device))
        tf[torch.from_numpy(nbrs == 0)] = 0.0
        return nf, ef, tf

    def set_neighbor_sampler(self, sampler):
        self.neighbor_sampler = sampler
        if sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            sampler.reset_random_state()

    def get_learned_scales(self):
        return [{'layer': i, **l.get_scale_statistics()} for i, l in enumerate(self.wavelet_transformers)]
