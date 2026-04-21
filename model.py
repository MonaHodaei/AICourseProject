import time
import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 transpose=False, bn=True, activation_fn=None):
        super(SharedMLP, self).__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=(kernel_size - 1) // 2)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=(kernel_size - 1) // 2)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    """lse1: encode_pos=True  → 10-dim relative position encoding.
       lse2: encode_pos=False → reuse relative_features from lse1."""

    def __init__(self, dim_in, dim_out, num_neighbors, encode_pos=False):
        super(LocalSpatialEncoding, self).__init__()
        self.num_neighbors = num_neighbors
        self.encode_pos    = encode_pos
        self.mlp = SharedMLP(dim_in, dim_out, activation_fn=nn.LeakyReLU(0.2))

    def gather_neighbor(self, coords, neighbor_indices):
        B, N, K = neighbor_indices.size()
        dim     = coords.shape[2]
        extended_indices = neighbor_indices.unsqueeze(1).expand(B, dim, N, K)
        extended_coords  = coords.transpose(-2, -1).unsqueeze(-1).expand(B, dim, N, K)
        return torch.gather(extended_coords, 2, extended_indices)

    def forward(self, coords, features, neighbor_indices, relative_features=None):
        B, N, K = neighbor_indices.size()

        if self.encode_pos:
            neighbor_coords  = self.gather_neighbor(coords, neighbor_indices)
            extended_coords  = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
            relative_pos     = extended_coords - neighbor_coords
            relative_dist    = torch.sqrt(
                torch.sum(torch.square(relative_pos), dim=1, keepdim=True))
            relative_features = torch.cat(
                [relative_dist, relative_pos, extended_coords, neighbor_coords], dim=1)
        else:
            if relative_features is None:
                raise ValueError("lse2 requires relative_features from lse1.")

        relative_features = self.mlp(relative_features)

        neighbor_features = self.gather_neighbor(
            features.transpose(1, 2).squeeze(3), neighbor_indices)

        return torch.cat([neighbor_features, relative_features], dim=1), relative_features


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels),   # bias=True by default
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, activation_fn=nn.LeakyReLU(0.2))

    def forward(self, x):
        scores   = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = torch.sum(scores * x, dim=-1, keepdim=True)
        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()
        self.num_neighbors = num_neighbors
        self.mlp1     = SharedMLP(d_in,   d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.lse1     = LocalSpatialEncoding(10,       d_out//2, num_neighbors, encode_pos=True)
        self.pool1    = AttentivePooling(d_out,        d_out//2)
        self.lse2     = LocalSpatialEncoding(d_out//2, d_out//2, num_neighbors, encode_pos=False)
        self.pool2    = AttentivePooling(d_out,        d_out)
        self.mlp2     = SharedMLP(d_out,   2*d_out)
        self.shortcut = SharedMLP(d_in,    2*d_out)
        self.lrelu    = nn.LeakyReLU()

    def forward(self, coords, feat, neighbor_indices):
        x = self.mlp1(feat)
        x, neighbor_features = self.lse1(coords, x, neighbor_indices)
        x = self.pool1(x)
        x, _ = self.lse2(coords, x, neighbor_indices, relative_features=neighbor_features)
        x = self.pool2(x)
        return self.lrelu(self.mlp2(x) + self.shortcut(feat))


class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4,
                 device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        self.num_neighbors = num_neighbors
        self.decimation    = decimation
        self.device        = device

        # ── Input projection ──────────────────────────────────────────
        # fc0.weight [8, 6], bn0 is plain BatchNorm2d
        self.fc0  = nn.Linear(d_in, 8)
        self.bn0  = nn.BatchNorm2d(8, eps=1e-6, momentum=0.99)
        self.lrelu0 = nn.LeakyReLU(0.2)

        # ── Encoder ───────────────────────────────────────────────────
        # enc.0: d_in=8,   d_out=16  → out=32
        # enc.1: d_in=32,  d_out=64  → out=128
        # enc.2: d_in=128, d_out=128 → out=256
        # enc.3: d_in=256, d_out=256 → out=512
        # enc.4: d_in=512, d_out=512 → out=1024
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8,   16,  num_neighbors),
            LocalFeatureAggregation(32,  64,  num_neighbors),
            LocalFeatureAggregation(128, 128, num_neighbors),
            LocalFeatureAggregation(256, 256, num_neighbors),
            LocalFeatureAggregation(512, 512, num_neighbors),
        ])

        # ── Bottleneck: 1024→1024 with bn ─────────────────────────────
        self.mlp = SharedMLP(1024, 1024, activation_fn=nn.LeakyReLU(0.2))

        # ── Decoder ───────────────────────────────────────────────────
        # encoder_dim_list (built like original code):
        # after enc.0: append 32, append 32   → [32, 32]
        # after enc.1: append 128             → [32, 32, 128]
        # after enc.2: append 256             → [32, 32, 128, 256]
        # after enc.3: append 512             → [32, 32, 128, 256, 512]
        # after enc.4: append 1024            → [32, 32, 128, 256, 512, 1024]
        #
        # decoder input = encoder_dim_list[-i-2] + dim_feature
        # dec.0: 512  + 1024 = 1536 → 512
        # dec.1: 256  + 512  = 768  → 256
        # dec.2: 128  + 256  = 384  → 128
        # dec.3: 32   + 128  = 160  → 32
        # dec.4: 32   + 32   = 64   → 32
        self.decoder = nn.ModuleList([
            SharedMLP(1536, 512, transpose=True, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(768,  256, transpose=True, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(384,  128, transpose=True, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(160,  32,  transpose=True, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64,   32,  transpose=True, activation_fn=nn.LeakyReLU(0.2)),
        ])

        # ── Head: fc1.0(32→64), fc1.1(64→32), dropout, fc1.3(32→C) ──
        self.fc1 = nn.Sequential(
            SharedMLP(32, 64, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2)),
            nn.Dropout(0.5),
            SharedMLP(32, num_classes, bn=False)
        )

        self = self.to(device)

    @staticmethod
    def random_sample(feature, pool_idx):
        """Subsample features using pool_idx."""
        feature   = feature.squeeze(3)
        num_neigh = pool_idx.size()[2]
        B         = feature.size()[0]
        d         = feature.size()[1]
        pool_idx  = torch.reshape(pool_idx, (B, -1))
        pool_idx  = pool_idx.unsqueeze(2).expand(B, -1, d)
        feature   = feature.transpose(1, 2)
        pool_features = torch.gather(feature, 1, pool_idx)
        pool_features = torch.reshape(pool_features, (B, -1, num_neigh, d))
        pool_features, _ = torch.max(pool_features, 2, keepdim=True)
        return pool_features.permute(0, 3, 1, 2)

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """Upsample features using nearest neighbor indices."""
        feature        = feature.squeeze(3)
        d              = feature.size(1)
        B              = interp_idx.size()[0]
        up_num_points  = interp_idx.size()[1]
        interp_idx     = torch.reshape(interp_idx, (B, up_num_points))
        interp_idx     = interp_idx.unsqueeze(1).expand(B, d, -1)
        interpolated   = torch.gather(feature, 2, interp_idx)
        return interpolated.unsqueeze(3)

    def forward(self, inputs):
        """
        Args:
            inputs: dict with keys:
                'features'        : (B, N, d_in)
                'coords'          : list of (B, N_i, 3) per layer
                'neighbor_indices': list of (B, N_i, K) per layer
                'sub_idx'         : list of (B, N_i//r, K) per layer
                'interp_idx'      : list of (B, N_i, 1) per layer
        """
        feat      = inputs['features'].to(self.device)
        coords    = [c.to(self.device) for c in inputs['coords']]
        neighbors = [n.to(self.device) for n in inputs['neighbor_indices']]
        sub_idx   = [s.to(self.device) for s in inputs['sub_idx']]
        interp_idx = [i.to(self.device) for i in inputs['interp_idx']]

        # Input projection
        feat = self.lrelu0(self.bn0(
            self.fc0(feat).transpose(-2, -1).unsqueeze(-1)
        ))

        # ── Encoder ───────────────────────────────────────────────────
        encoder_feat_list = []
        for i, lfa in enumerate(self.encoder):
            feat = lfa(coords[i], feat, neighbors[i])
            feat_sampled = self.random_sample(feat, sub_idx[i])
            if i == 0:
                encoder_feat_list.append(feat.clone())
            encoder_feat_list.append(feat_sampled.clone())
            feat = feat_sampled

        # ── Bottleneck ────────────────────────────────────────────────
        feat = self.mlp(feat)

        # ── Decoder ───────────────────────────────────────────────────
        for i, dec in enumerate(self.decoder):
            feat_interp = self.nearest_interpolation(feat, interp_idx[-i - 1])
            feat = dec(torch.cat([encoder_feat_list[-i - 2], feat_interp], dim=1))

        # ── Head ──────────────────────────────────────────────────────
        scores = self.fc1(feat)
        return scores.squeeze(3).transpose(1, 2)   # (B, N, num_classes)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Quick shape test  inputs
    B, N, d_in = 1, 45056, 6
    num_layers  = 5
    sub_ratio   = 4
    K           = 16

    dummy_inputs = {
        'features': torch.randn(B, N, d_in),
        'coords':   [],
        'neighbor_indices': [],
        'sub_idx':  [],
        'interp_idx': [],
    }

    n = N
    for i in range(num_layers):
        dummy_inputs['coords'].append(torch.randn(B, n, 3))
        dummy_inputs['neighbor_indices'].append(
            torch.randint(0, n, (B, n, K), dtype=torch.int64))
        n_sub = n // sub_ratio
        dummy_inputs['sub_idx'].append(
            torch.randint(0, n, (B, n_sub, K), dtype=torch.int64))
        dummy_inputs['interp_idx'].append(
            torch.randint(0, n_sub, (B, n, 1), dtype=torch.int64))
        n = n_sub

    model = RandLANet(d_in, 8, K, sub_ratio, device)
    model.eval()
    t0  = time.time()
    out = model(dummy_inputs)
    t1  = time.time()
    print(f'Output shape: {out.shape}  |  Time: {t1-t0:.2f}s')