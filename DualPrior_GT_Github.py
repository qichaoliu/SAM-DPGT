"""
DualPriorGT_Github: Dual-Prior Guided Graph Transformer for Few-Shot HSI Classification.

Github-ready version (simplified from DualPriorGTSimple):
- No hierarchical pooling/unpooling
- No edgepooling node merging
- Single superpixel graph (SAM-guided)
- L stacked DPAttn blocks with constant dimension (L=1 in experiments)
- No spectral edge pruning (raw spatial adjacency)
- No pixel-to-node linear projection (direct pooling into DPAttn)
- Skip connection: transformer output upsampled + original pixel features -> CNN Tail

Core innovations:
1. Spatial Geometric Prior: RBF distance bias injected into attention scores.
2. Head Diversification: Temperature scaling prevents attention collapse.
3. k-hop connectivity mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = torch.device("cuda:0")


class SpatialBias(nn.Module):
    """Geometric prior: precomputed RBF distance bias."""
    def __init__(self, n_rbf_centers: int = 16, max_dist: float = 50.0):
        super().__init__()
        self.n_rbf_centers = n_rbf_centers
        rbf_centers = torch.linspace(0, max_dist, n_rbf_centers)
        rbf_sigma = max_dist / n_rbf_centers
        rbf_weights = torch.exp(-torch.arange(n_rbf_centers).float() * 0.5)
        self.register_buffer('rbf_centers', rbf_centers)
        self.register_buffer('rbf_sigma', torch.tensor(rbf_sigma))
        self.register_buffer('rbf_weights', rbf_weights)

    def compute(self, pos: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(pos, pos)
        sigma = self.rbf_sigma + 1e-6
        dist_sq = dist.unsqueeze(-1)
        rbf_centers = self.rbf_centers.to(pos.device)
        rbf = torch.exp(-((dist_sq - rbf_centers) ** 2) / (2 * sigma ** 2))
        return torch.einsum('ijc,c->ij', rbf, self.rbf_weights.to(pos.device))


class DPAttn(nn.Module):
    """Dual-Prior Attention Layer.
    
    Architecture:
      H_shared = W_shared(H)
      Q, K, V = split heads from H_shared
      e = Q @ K^T / scale
      A = softmax((e + mask * alpha_s * spatial) / temp)
      attn_out = A @ V
      out = activation(W_out(attn_out))
    """
    def __init__(self, n_features, num_heads=4, activation=None, n_nodes=None):
        super().__init__()
        assert n_features % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = n_features // num_heads
        self.activation = activation

        # Shared projection
        self.W_shared = nn.Linear(n_features, n_features)
        
        # Output projection
        self.W_out = nn.Linear(n_features, n_features)

        # Per-head learnable fusion weight for spatial prior
        self.alpha_s = nn.Parameter(torch.ones(num_heads) * 2.0)

        # Head temperature for diversification
        self.head_temperature = nn.Parameter(torch.ones(num_heads))

        # Per-layer learnable self-loop weight (initial value = 1.0)
        self.self_loop_weight = nn.Parameter(torch.ones(1))

    def forward(self, X, mask, s_bias_buf):
        N = X.shape[0]
        mask = mask.to(X.device)
        s_bias_buf = s_bias_buf.to(X.device)
        
        # Shared projection
        H_shared = self.W_shared(X)  # [N, d]
        
        # Split into heads
        Q = H_shared.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = H_shared.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = H_shared.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # Batched attention scores: [num_heads, N, N]
        scale = math.sqrt(self.head_dim)
        e = torch.einsum('hqd,hkd->hqk', Q, K) / scale

        # Add learnable self-loop weight to diagonal
        diag_mask = torch.eye(N, device=X.device).unsqueeze(0).expand(self.num_heads, -1, -1)
        e = e + self.self_loop_weight * diag_mask

        # Add spatial prior
        alpha_s = F.softplus(self.alpha_s).view(self.num_heads, 1, 1)
        spatial_term = alpha_s * s_bias_buf.unsqueeze(0)

        # Apply head temperature
        temp = self.head_temperature.to(X.device).view(self.num_heads, 1, 1)
        A = (e + mask.unsqueeze(0) * spatial_term) / (temp + 0.1)

        # Mask & softmax
        zero_vec = -9e15 * torch.ones_like(A)
        A = torch.where(mask.unsqueeze(0) > 0, A, zero_vec)
        A_softmax = F.softmax(A, dim=2)

        # Attention output
        attn_out = torch.bmm(A_softmax, V)
        attn_out = attn_out.transpose(0, 1).contiguous()
        attn_out = attn_out.view(N, -1)
        out = self.W_out(attn_out)
        
        return self.activation(out) if self.activation else out


class DualPriorGT_Github(nn.Module):
    """
    DualPriorGT_Github - Github-ready version.
    
    Architecture:
      1. CNN Head (1x1 conv): pixel-level feature extraction
      2. Pixel-to-superpixel pooling via SAM assignment (direct, no linear projection)
      3. L stacked DPAttn blocks (constant dimension, same graph)
      4. Superpixel-to-pixel upsampling
      5. Skip connection: upsampled + original pixel features -> CNN Tail
      6. Classifier
    """
    def __init__(self, height: int, width: int, channels: int, class_count: int,
                 n_superpixels: int, adjacency: torch.Tensor,
                 node_positions: torch.Tensor,
                 num_blocks: int = 1, k_hop: int = 2,
                 num_heads: int = 4, base_channels: int = 128,
                 use_rbf: bool = True):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.class_count = class_count
        self.n_superpixels = n_superpixels
        self.num_blocks = num_blocks
        self.k_hop = k_hop
        self.num_heads = num_heads
        self.base_channels = base_channels
        self.use_rbf = use_rbf
        self.n_pixels = height * width

        # Activation
        self.act = nn.LeakyReLU()

        # === Graph construction ===
        # Raw spatial adjacency (no spectral pruning)
        A = adjacency.to(device).float()
        
        # k-hop connectivity
        A_with_self = A + torch.eye(A.shape[0], device=device)
        A_k = A_with_self.clone()
        for _ in range(self.k_hop - 1):
            A_k = A_k @ A_with_self
        self.register_buffer('mask', (A_k > 0.5).float())

        # Node positions and RBF spatial bias
        pos = node_positions.to(device).float()
        if pos.shape[0] > 1:
            max_d = torch.max(torch.norm(pos.unsqueeze(1) - pos.unsqueeze(0), dim=-1)).item()
        else:
            max_d = 1.0
        max_d = max(max_d, 1.0)
        
        if self.use_rbf:
            sp_bias = SpatialBias(16, max_d)
            s_bias = sp_bias.compute(pos)
        else:
            s_bias = torch.zeros(pos.shape[0], pos.shape[0], device=device)
        self.register_buffer('s_bias_buf', s_bias)

        # === Pixel-to-superpixel assignment ===
        self.register_buffer('S_hat_T', torch.zeros(n_superpixels, self.n_pixels))
        self.register_buffer('S_T', torch.zeros(self.n_pixels, n_superpixels))

        # === CNN Head (1x1 conv only) ===
        self.CNN_head = nn.Sequential(
            nn.Conv2d(self.channels, self.base_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.base_channels),
            nn.LeakyReLU()
        )

        # === Projection: pixel features -> superpixel node features ===
        self.pixel_to_node = nn.Linear(self.base_channels, self.base_channels)

        # === L stacked DPAttn blocks (constant dimension) ===
        self.attn_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.attn_blocks.append(DPAttn(
                self.base_channels, num_heads=self.num_heads,
                activation=self.act, n_nodes=n_superpixels
            ))

        # === CNN Tail ===
        tail_in_ch = self.base_channels + self.base_channels
        self.CNN_tail = nn.Sequential(
            nn.BatchNorm2d(tail_in_ch),
            nn.Conv2d(tail_in_ch, self.base_channels, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.base_channels, self.base_channels,
                      kernel_size=5, stride=1, padding=2, groups=self.base_channels),
            nn.LeakyReLU()
        )

        # === Classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(self.base_channels, self.class_count),
            nn.Softmax(dim=-1)
        )

    def set_assignment_matrices(self, S_hat_T: torch.Tensor, S_T: torch.Tensor):
        """Set pixel-superpixel assignment matrices."""
        self.S_hat_T.copy_(S_hat_T.to(device))
        self.S_T.copy_(S_T.to(device))

    def forward(self, x: torch.Tensor):
        """
        x: [H, W, C] raw spectral data
        Returns: logits, pixel_features, [final_features]
        """
        h, w, c = x.shape

        # CNN Head (1x1 conv)
        x = torch.unsqueeze(x.permute([2, 0, 1]), 0)
        H_0 = self.CNN_head(x)
        H_0 = torch.squeeze(H_0, 0).permute([1, 2, 0])  # [H, W, base_channels]
        H_pixel = H_0.reshape([h * w, -1])  # [H*W, base_channels] - save for skip

        # Pixel-to-superpixel pooling
        H_node = torch.mm(self.S_hat_T, H_pixel)  # [N, base_channels]

        # Project to target dimension
        H_node = self.pixel_to_node(H_node)
        H_node = self.act(H_node)

        # L stacked DPAttn blocks
        for block in self.attn_blocks:
            H_node = block(H_node, self.mask, self.s_bias_buf)

        # Superpixel-to-pixel upsampling
        H_upsampled = torch.mm(self.S_T, H_node)  # [H*W, base_channels]

        # Skip connection: upsampled + original pixel features
        H_combined = torch.cat([H_upsampled, H_pixel], dim=-1)

        # CNN Tail
        H_2d = H_combined.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
        H_out = self.CNN_tail(H_2d)
        H_out = H_out.squeeze(0).permute(1, 2, 0).reshape(h * w, -1)

        # Classification
        logits = self.classifier(H_out)
        return logits, H_pixel, [H_out]
