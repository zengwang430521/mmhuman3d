# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import OrderedDict
from mmcv.cnn import (
    build_norm_layer, constant_init, normal_init,
    trunc_normal_init, build_conv_layer)
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import _load_checkpoint, load_state_dict
from mmcv.runner.base_module import BaseModule
from mmcv.utils import get_logger, to_2tuple

try:
    from torch.cuda.amp import autocast
    WITH_AUTOCAST = True
except ImportError:
    WITH_AUTOCAST = False


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use `get_logger` method in mmcv to get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmpose".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


def get_grid_index(init_grid_size, map_size, device):
    """For every initial grid, get its index in the feature map.
    Note:
        [H_init, W_init]: shape of initial grid
        [H, W]: shape of feature map
        N_init: numbers of initial token

    Args:
        init_grid_size (list[int] or tuple[int]): initial grid resolution in
            format [H_init, W_init].
        map_size (list[int] or tuple[int]): feature map resolution in format
            [H, W].
        device: the device of output

    Returns:
        idx (torch.LongTensor[B, N_init]): index in flattened feature map.
    """
    H_init, W_init = init_grid_size
    H, W = map_size
    idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
    idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
    return idx.flatten()


def index_points(points, idx):
    """Sample features following the index.
    Note:
        B: batch size
        N: point number
        C: channel number of each point
        Ns: sampled point number

    Args:
        points (torch.Tensor[B, N, C]): input points data
        idx (torch.LongTensor[B, S]): sample index

    Returns:
        new_points (torch.Tensor[B, Ns, C]):, indexed points data
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(
        B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def token2map(token_dict):
    """Transform vision tokens to feature map. This function only works when
    the resolution of the feature map is not higher than the initial grid
    structure.

    Note:
        B: batch size
        C: channel number of each token
        [H, W]: shape of feature map
        N_init: numbers of initial token

    Args:
        token_dict (dict): dict for token information.

    Returns:
        x_out (Tensor[B, C, H, W]): feature map.
    """

    x = token_dict['x']
    H, W = token_dict['map_size']
    H_init, W_init = token_dict['init_grid_size']
    idx_token = token_dict['idx_token']
    B, N, C = x.shape
    N_init = H_init * W_init
    device = x.device

    if N_init == N and N == H * W:
        # for the initial tokens with grid structure, just reshape
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    # for each initial grid, get the corresponding index in
    # the flattened feature map.
    idx_hw = get_grid_index([H_init, W_init], [H, W],
                            device=device)[None, :].expand(B, -1)
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    value = x.new_ones(B * N_init)

    # choose the way with fewer flops.
    if N_init < N * H * W:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_hw = idx_hw + idx_batch * H * W
        idx_tokens = idx_token + idx_batch * N
        coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)

        # torch.sparse do not support gradient for
        # sparse tensor, so we detach it
        value = value.detach().to(torch.float32)

        # build a sparse matrix with the shape [B * H * W, B * N]
        A = torch.sparse.FloatTensor(coor, value,
                                     torch.Size([B * H * W, B * N]))

        # normalize the weight for each row
        if WITH_AUTOCAST:
            with autocast(enabled=False):
                all_weight = A @ x.new_ones(B * N, 1).type(
                    torch.float32) + 1e-6
        else:
            all_weight = A @ x.new_ones(B * N, 1).type(torch.float32) + 1e-6
        value = value / all_weight[idx_hw.reshape(-1), 0]

        # update the matrix with normalize weight
        A = torch.sparse.FloatTensor(coor, value,
                                     torch.Size([B * H * W, B * N]))

        # sparse matrix multiplication
        if WITH_AUTOCAST:
            with autocast(enabled=False):
                x_out = A @ x.reshape(B * N, C).to(torch.float32)  # [B*H*W, C]
        else:
            x_out = A @ x.reshape(B * N, C).to(torch.float32)  # [B*H*W, C]

    else:
        # use dense matrix multiplication
        # Flops: B * N * H * W * (C+2)
        coor = torch.stack([idx_batch, idx_hw, idx_token],
                           dim=0).reshape(3, B * N_init)

        # build a matrix with shape [B, H*W, N]
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W,
                                                              N])).to_dense()
        # normalize the weight
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        x_out = A @ x  # [B, H*W, C]

    x_out = x_out.type(x.dtype)
    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return x_out


def map2token(feature_map, token_dict):
    """Transform feature map to vision tokens. This function only works when
    the resolution of the feature map is not higher than the initial grid
    structure.

    Note:
        B: batch size
        C: channel number
        [H, W]: shape of feature map
        N_init: numbers of initial token

    Args:
        feature_map (Tensor[B, C, H, W]): feature map.
        token_dict (dict): dict for token information.

    Returns:
        out (Tensor[B, N, C]): token features.
    """
    idx_token = token_dict['idx_token']
    N = token_dict['token_num']
    H_init, W_init = token_dict['init_grid_size']
    N_init = H_init * W_init

    B, C, H, W = feature_map.shape
    device = feature_map.device

    if N_init == N and N == H * W:
        # for the initial tokens with grid structure, just reshape
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    idx_hw = get_grid_index([H_init, W_init], [H, W],
                            device=device)[None, :].expand(B, -1)

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    value = feature_map.new_ones(B * N_init)

    # choose the way with fewer flops.
    if N_init < N * H * W:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_token = idx_token + idx_batch * N
        idx_hw = idx_hw + idx_batch * H * W
        indices = torch.stack([idx_token, idx_hw], dim=0).reshape(2, -1)

        # sparse mm do not support gradient for sparse matrix
        value = value.detach().to(torch.float32)
        # build a sparse matrix with shape [B*N, B*H*W]
        A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
        # normalize the matrix
        if WITH_AUTOCAST:
            with autocast(enabled=False):
                all_weight = A @ torch.ones(
                    [B * H * W, 1], device=device, dtype=torch.float32) + 1e-6
        else:
            all_weight = A @ torch.ones(
                [B * H * W, 1], device=device, dtype=torch.float32) + 1e-6
        value = value / all_weight[idx_token.reshape(-1), 0]

        A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
        # out: [B*N, C]
        if WITH_AUTOCAST:
            with autocast(enabled=False):
                out = A @ feature_map.permute(0, 2, 3, 1).contiguous().reshape(
                    B * H * W, C).float()
        else:
            out = A @ feature_map.permute(0, 2, 3, 1).contiguous().reshape(
                B * H * W, C).float()
    else:
        # use dense matrix multiplication
        # Flops: B * N * H * W * (C+2)
        indices = torch.stack([idx_batch, idx_token, idx_hw],
                              dim=0).reshape(3, -1)
        value = value.detach()  # To reduce the training time, we detach here.
        A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
        # normalize the matrix
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W,
                                                          C).contiguous()

    out = out.type(feature_map.dtype)
    out = out.reshape(B, N, C)
    return out


def token_interp(target_dict, source_dict):
    """Transform token features between different distribution.

    Note:
        B: batch size
        N: token number
        C: channel number

    Args:
        target_dict (dict): dict for target token information
        source_dict (dict): dict for source token information.

    Returns:
        x_out (Tensor[B, N, C]): token features.
    """

    x_s = source_dict['x']
    idx_token_s = source_dict['idx_token']
    idx_token_t = target_dict['idx_token']
    T = target_dict['token_num']
    B, S, C = x_s.shape
    N_init = idx_token_s.shape[1]

    weight = target_dict['agg_weight'] if 'agg_weight' in target_dict.keys(
    ) else None
    if weight is None:
        weight = x_s.new_ones(B, N_init, 1)
    weight = weight.reshape(-1)

    # choose the way with fewer flops.
    if N_init < T * S:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_token_t = idx_token_t + torch.arange(
            B, device=x_s.device)[:, None] * T
        idx_token_s = idx_token_s + torch.arange(
            B, device=x_s.device)[:, None] * S
        coor = torch.stack([idx_token_t, idx_token_s],
                           dim=0).reshape(2, B * N_init)

        # torch.sparse does not support grad for sparse matrix
        weight = weight.float().detach().to(torch.float32)
        # build a matrix with shape [B*T, B*S]
        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B * T, B * S]))
        # normalize the matrix
        if WITH_AUTOCAST:
            with autocast(enabled=False):
                all_weight = A.type(torch.float32) @ x_s.new_ones(
                    B * S, 1).type(torch.float32) + 1e-6
        else:
            all_weight = A.type(torch.float32) @ x_s.new_ones(B * S, 1).type(
                torch.float32) + 1e-6
        weight = weight / all_weight[idx_token_t.reshape(-1), 0]
        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B * T, B * S]))
        # sparse matmul
        if WITH_AUTOCAST:
            with autocast(enabled=False):
                x_out = A.type(torch.float32) @ x_s.reshape(B * S, C).type(
                    torch.float32)
        else:
            x_out = A.type(torch.float32) @ x_s.reshape(B * S, C).type(
                torch.float32)
    else:
        # use dense matrix multiplication
        # Flops: B * T * S * (C+2)
        idx_batch = torch.arange(
            B, device=x_s.device)[:, None].expand(B, N_init)
        coor = torch.stack([idx_batch, idx_token_t, idx_token_s],
                           dim=0).reshape(3, B * N_init)
        weight = weight.detach()  # detach to reduce training time
        # build a matrix with shape [B, T, S]
        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B, T,
                                                               S])).to_dense()
        # normalize the matrix
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        # dense matmul
        x_out = A @ x_s

    x_out = x_out.reshape(B, T, C).type(x_s.dtype)
    return x_out


def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.

    Note:
        B: batch size
        N: token number
        C: channel number

    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicating which token is the
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.

    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. In this function, it equals
            to the input cluster number.
    """

    with torch.no_grad():
        x = token_dict['x']
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C**0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the
            # distance between empty tokens and any other
            # tokens should be the maximal distance.
            dist_matrix = \
                dist_matrix * token_mask[:, None, :] +\
                (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(
            dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest**2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max *
                              (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(
            B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(
            cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1),
                    index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster. Implemented by
    torch.index_add(). Flops: B*N*(C+2)

    Note:
        B: batch size
        N: token number
        C: channel number

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.

    Return:
        out_dict (dict): dict for output token information
    """

    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(
        dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(
        dim=0,
        index=idx.reshape(B * N),
        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['map_size'] = token_dict['map_size']
    out_dict['init_grid_size'] = token_dict['init_grid_size']
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    return out_dict


class MLP(nn.Module):
    """FFN with Depthwise Conv of TCFormer.

    Args:
        in_features (int): The feature dimension.
        hidden_features (int, optional): The hidden dimension of FFNs.
            Defaults: The same as in_features.
        out_features (int, optional): The output feature dimension.
            Defaults: The same as in_features.
        act_layer (nn.Module, optional): The activation config for FFNs.
            Default: nn.GELU.
        drop (float, optional): drop out rate. Default: 0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def init_weights(self):
        """init weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """Depthwise Conv for regular grid-based tokens.

    Args:
        dim (int): The feature dimension.
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TCFormerRegularAttention(nn.Module):
    """Spatial Reduction Attention for regular grid-based tokens.

    Args:
        dim (int): The feature dimension of tokens,
        num_heads (int): Parallel attention heads.
        qkv_bias (bool): enable bias for qkv if True. Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after attention process.
            Default: 0.0.
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention. Default: 1.
        use_sr_conv (bool): If True, use a conv layer for spatial reduction.
            If False, use a pooling process for spatial reduction. Defaults:
            True.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        sr_ratio=1,
        use_sr_conv=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, \
            f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.use_sr_conv = use_sr_conv
        if sr_ratio > 1 and self.use_sr_conv:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            kv = x.permute(0, 2, 1).reshape(B, C, H, W)
            if self.use_sr_conv:
                kv = self.sr(kv).reshape(B, C, -1).permute(0, 2,
                                                           1).contiguous()
                kv = self.norm(kv)
            else:
                kv = F.avg_pool2d(
                    kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        else:
            kv = x

        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads,
                                 C // self.num_heads).permute(2, 0, 3, 1,
                                                              4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TCFormerRegularBlock(nn.Module):
    """Transformer block for regular grid-based tokens.

    Args:
        dim (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratio (int): The expansion ratio for the FFNs.
        qkv_bias (bool): enable bias for qkv if True. Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float): Dropout layers after attention process and in FFN.
            Default: 0.0.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        drop_path (int, optional): The drop path rate of transformer block.
            Default: 0.0
        act_layer (nn.Module, optional): The activation config for FFNs.
            Default: nn.GELU.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention. Default: 1.
        use_sr_conv (bool): If True, use a conv layer for spatial reduction.
            If False, use a pooling process for spatial reduction. Defaults:
            True.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 use_sr_conv=True):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]

        self.attn = TCFormerRegularAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            use_sr_conv=use_sr_conv)
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path))

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class TokenConv(nn.Conv2d):
    """Conv layer for dynamic tokens.

    A skip link is added between the input and output tokens to reserve detail
    tokens.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(
            in_channels=kwargs['in_channels'],
            out_channels=kwargs['out_channels'],
            kernel_size=1,
            bias=False,
            groups=groups)

    def forward(self, token_dict):
        x = token_dict['x']
        x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_map = token2map(token_dict)
        x_map = super().forward(x_map)
        x = x + map2token(x_map, token_dict)
        return x


class TCMLP(nn.Module):
    """FFN with Depthwise Conv for dynamic tokens.

    Args:
        in_features (int): The feature dimension.
        hidden_features (int, optional): The hidden dimension of FFNs.
            Defaults: The same as in_features.
        out_features (int, optional): The output feature dimension.
            Defaults: The same as in_features.
        act_layer (nn.Module, optional): The activation config for FFNs.
            Default: nn.GELU.
        drop (float, optional): drop out rate. Default: 0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = TokenConv(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def init_weights(self):
        """init weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, token_dict):
        token_dict['x'] = self.fc1(token_dict['x'])
        x = self.dwconv(token_dict)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TCFormerDynamicAttention(TCFormerRegularAttention):
    """Spatial Reduction Attention for dynamic tokens."""

    def forward(self, q_dict, kv_dict):
        """Attention process for dynamic tokens.
        Dynamic tokens are represented by a dict with the following keys:
            x (torch.Tensor[B, N, C]): token features.
            token_num(int): token number.
            map_size(list[int] or tuple[int]): feature map resolution in
                format [H, W].
            init_grid_size(list[int] or tuple[int]): initial grid resolution
                in format [H_init, W_init].
            idx_token(torch.LongTensor[B, N_init]): indicates which token
                the initial grid belongs to.
            agg_weight(torch.LongTensor[B, N_init] or None): weight for
                aggregation. Indicates the weight of each token in its
                cluster. If set to None, uniform weight is used.

        Note:
            B: batch size
            N: token number
            C: channel number
            Ns: sampled point number
            [H_init, W_init]: shape of initial grid
            [H, W]: shape of feature map
            N_init: numbers of initial token

        Args:
            q_dict (dict): dict for query token information
            kv_dict (dict): dict for key and value token information

        Return:
            x (torch.Tensor[B, N, C]): output token features.
        """

        q = q_dict['x']
        kv = kv_dict['x']
        B, Nq, C = q.shape
        Nkv = kv.shape[1]
        conf_kv = kv_dict['token_score'] if 'token_score' in kv_dict.keys(
        ) else kv.new_zeros(B, Nkv, 1)

        q = self.q(q).reshape(B, Nq, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            tmp = torch.cat([kv, conf_kv], dim=-1)
            tmp_dict = kv_dict.copy()
            tmp_dict['x'] = tmp
            tmp_dict['map_size'] = q_dict['map_size']
            tmp = token2map(tmp_dict)

            kv = tmp[:, :C]
            conf_kv = tmp[:, C:]

            if self.use_sr_conv:
                kv = self.sr(kv)
                _, _, h, w = kv.shape
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()
                kv = self.norm(kv)
            else:
                kv = F.avg_pool2d(
                    kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()

            conf_kv = F.avg_pool2d(
                conf_kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            conf_kv = conf_kv.reshape(B, 1, -1).permute(0, 2, 1).contiguous()

        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads,
                                 C // self.num_heads).permute(2, 0, 3, 1,
                                                              4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        conf_kv = conf_kv.squeeze(-1)[:, None, None, :]
        attn = attn + conf_kv
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer block for dynamic tokens
class TCFormerDynamicBlock(TCFormerRegularBlock):
    """Transformer block for dynamic tokens.

    Args:
        dim (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratio (int): The expansion ratio for the FFNs.
        qkv_bias (bool): enable bias for qkv if True. Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float): Dropout layers after attention process and in FFN.
            Default: 0.0.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        drop_path (int, optional): The drop path rate of transformer block.
            Default: 0.0
        act_layer (nn.Module, optional): The activation config for FFNs.
            Default: nn.GELU.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention. Default: 1.
        use_sr_conv (bool): If True, use a conv layer for spatial reduction.
            If False, use a pooling process for spatial reduction. Defaults:
            True.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 use_sr_conv=True):
        super(TCFormerRegularBlock, self).__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]

        self.attn = TCFormerDynamicAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            use_sr_conv=use_sr_conv)
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path))

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TCMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (dict or tuple[dict] or list[dict]): input dynamic
                token information. If a single dict is provided, it's
                regraded as query and key, value. If a tuple or list
                of dict is provided, the first one is regarded as key
                and the second one is regarded as key, value.

        Return:
            q_dict (dict): dict for output token information
        """
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs
        else:
            q_dict, kv_dict = inputs, None

        x = q_dict['x']
        # norm1
        q_dict['x'] = self.norm1(q_dict['x'])
        if kv_dict is None:
            kv_dict = q_dict
        else:
            kv_dict['x'] = self.norm1(kv_dict['x'])

        # attn
        x = x + self.drop_path(self.attn(q_dict, kv_dict))

        # mlp
        q_dict['x'] = self.norm2(x)
        x = x + self.drop_path(self.mlp(q_dict))
        q_dict['x'] = x

        return q_dict


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        conv_type='Conv2d',
        kernel_size=16,
        stride=16,
        padding='corner',
        dilation=1,
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class CTM(nn.Module):
    """Clustering-based Token Merging module in TCFormer.

    Args:
        sample_ratio (float): The sample ratio of tokens.
        embed_dim (int): Input token feature dimension.
        dim_out (int): Output token feature dimension.
        k (int): number of the nearest neighbor used i DPC-knn algorithm.
    """

    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.conv = TokenConv(
            in_channels=embed_dim,
            out_channels=dim_out,
            kernel_size=3,
            stride=2,
            padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k

    def forward(self, token_dict):
        token_dict = token_dict.copy()
        x = self.conv(token_dict)
        x = self.norm(x)
        token_score = self.score(x)
        token_weight = token_score.exp()

        token_dict['x'] = x
        B, N, C = x.shape
        token_dict['token_score'] = token_score

        cluster_num = max(math.ceil(N * self.sample_ratio), 1)
        idx_cluster, cluster_num = cluster_dpc_knn(token_dict, cluster_num,
                                                   self.k)
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num,
                                 token_weight)

        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        down_dict['map_size'] = [H, W]

        return down_dict, token_dict


def tcformer_convert(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt.items():
        if 'patch_embed' in k:
            new_k = k.replace('.proj.', '.projection.')
        else:
            new_k = k
        new_ckpt[new_k] = v
    return new_ckpt


class TCFormer(nn.Module):
    """Token Clustering Transformer (TCFormer)

    Implementation of `Not All Tokens Are Equal: Human-centric Visual
    Analysis via Token Clustering Transformer
    <https://arxiv.org/abs/2204.08680>`

        Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (list[int]): Embedding dimension. Default:
            [64, 128, 256, 512].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer block.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN', eps=1e-6).
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer block. Default: [8, 4, 2, 1].
        num_stages (int): The num of stages. Default: 4.
        pretrained (str, optional): model pretrained path. Default: None.
        k (int): number of the nearest neighbor used for local density.
        sample_ratios (list[float]): The sample ratios of CTM modules.
            Default: [0.25, 0.25, 0.25]
        return_map (bool): If True, transfer dynamic tokens to feature map at
            last. Default: False
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 num_layers=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 num_stages=4,
                 pretrained=None,
                 k=5,
                 sample_ratios=[0.25, 0.25, 0.25],
                 return_map=False,
                 convert_weights=True):
        super().__init__()

        self.num_layers = num_layers
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.convert_weights = convert_weights

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]
        cur = 0

        # In stage 1, use the standard transformer blocks
        for i in range(1):
            patch_embed = PatchEmbed(
                in_channels=in_channels if i == 0 else embed_dims[i - 1],
                embed_dims=embed_dims[i],
                kernel_size=7,
                stride=4,
                padding=3,
                bias=True,
                norm_cfg=dict(type='LN', eps=1e-6))

            block = nn.ModuleList([
                TCFormerRegularBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i]) for j in range(num_layers[i])
            ])
            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]

            cur += num_layers[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

        # In stage 2~4, use TCFormerDynamicBlock for dynamic tokens
        for i in range(1, num_stages):
            ctm = CTM(sample_ratios[i - 1], embed_dims[i - 1], embed_dims[i],
                      k)

            block = nn.ModuleList([
                TCFormerDynamicBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i]) for j in range(num_layers[i])
            ])
            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            cur += num_layers[i]

            setattr(self, f'ctm{i}', ctm)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()

            checkpoint = _load_checkpoint(
                pretrained, logger=logger, map_location='cpu')
            logger.warning(f'Load pre-trained model for '
                           f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if self.convert_weights:
                # We need to convert pre-trained weights to match this
                # implementation.
                state_dict = tcformer_convert(state_dict)
            load_state_dict(self, state_dict, strict=False, logger=logger)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []

        i = 0
        patch_embed = getattr(self, f'patch_embed{i + 1}')
        block = getattr(self, f'block{i + 1}')
        norm = getattr(self, f'norm{i + 1}')
        x, (H, W) = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)

        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {
            'x': x,
            'token_num': N,
            'map_size': [H, W],
            'init_grid_size': [H, W],
            'idx_token': idx_token,
            'agg_weight': agg_weight
        }
        outs.append(token_dict.copy())

        # stage 2~4
        for i in range(1, self.num_stages):
            ctm = getattr(self, f'ctm{i}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            token_dict = ctm(token_dict)  # down sample
            for j, blk in enumerate(block):
                token_dict = blk(token_dict)

            token_dict['x'] = norm(token_dict['x'])
            outs.append(token_dict)

        if self.return_map:
            outs = [token2map(token_dict) for token_dict in outs]

        return outs
