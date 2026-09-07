import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(
            kernel_size=kernel_size, stride=stride, padding=padding)
        input_channels = reduce(lambda x, y: x * y, kernel_size) * channel
        self.embedding = nn.Linear(input_channels, hidden)

    def forward(self, value, batch_size, output_size):
        feature_h = int((output_size[0] + 2 * self.padding[0]
                         - (self.kernel_size[0] - 1) - 1)
                        / self.stride[0] + 1)
        feature_w = int((output_size[1] + 2 * self.padding[1]
                         - (self.kernel_size[1] - 1) - 1)
                        / self.stride[1] + 1)
        tokens = self.embedding(self.t2t(value).permute(0, 2, 1))
        return tokens.view(batch_size, -1, feature_h, feature_w, tokens.size(2))


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super().__init__()
        output_channels = reduce(lambda x, y: x * y, kernel_size) * channel
        self.embedding = nn.Linear(hidden, output_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, value, frame_count, output_size):
        batch_size, _, _, _, channels = value.shape
        tokens = self.embedding(value.reshape(batch_size, -1, channels))
        batch, _, flattened_channels = tokens.shape
        tokens = tokens.view(batch * frame_count, -1, flattened_channels)
        tokens = F.fold(
            tokens.permute(0, 2, 1),
            output_size=output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        return self.bias_conv(tokens)


class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1960, t2t_params=None):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        if t2t_params is None:
            raise ValueError('t2t_params is required')
        self.t2t_params = t2t_params
        self.kernel_shape = reduce(
            lambda x, y: x * y, t2t_params['kernel_size'])

    def forward(self, value, output_size):
        vector_count = 1
        for index, dimension in enumerate(self.t2t_params['kernel_size']):
            vector_count *= int(
                (output_size[index] + 2 * self.t2t_params['padding'][index]
                 - (dimension - 1) - 1)
                / self.t2t_params['stride'][index] + 1)

        value = self.fc1(value)
        batch, token_count, channels = value.shape
        normalizer = value.new_ones(
            batch, token_count, self.kernel_shape
        ).view(-1, vector_count, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(
            normalizer,
            output_size=output_size,
            kernel_size=self.t2t_params['kernel_size'],
            padding=self.t2t_params['padding'],
            stride=self.t2t_params['stride'],
        )
        value = F.fold(
            value.view(-1, vector_count, channels).permute(0, 2, 1),
            output_size=output_size,
            kernel_size=self.t2t_params['kernel_size'],
            padding=self.t2t_params['padding'],
            stride=self.t2t_params['stride'],
        )
        value = F.unfold(
            value / normalizer,
            kernel_size=self.t2t_params['kernel_size'],
            padding=self.t2t_params['padding'],
            stride=self.t2t_params['stride'],
        ).permute(0, 2, 1).contiguous().view(batch, token_count, channels)
        return self.fc2(value)


def window_partition(value, window_size, head_count):
    batch, frames, height, width, channels = value.shape
    value = value.view(
        batch,
        frames,
        height // window_size[0],
        window_size[0],
        width // window_size[1],
        window_size[1],
        head_count,
        channels // head_count,
    )
    return value.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()


class SparseWindowAttention(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size=(4, 4),
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 pooling_token=True):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError('dim must be divisible by n_head')
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if pooling_token:
            self.pool_layer = nn.Conv2d(
                dim, dim, kernel_size=pool_size, stride=pool_size,
                padding=0, groups=dim)
            self.pool_layer.weight.data.fill_(
                1.0 / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.zero_()
        self.expand_size = tuple((size + 1) // 2 for size in window_size)
        if any(size > 0 for size in self.expand_size):
            mask_tl = torch.ones(*window_size)
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(*window_size)
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(*window_size)
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(*window_size)
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            rolled_mask = torch.stack(
                (mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer(
                'valid_ind_rolled',
                rolled_mask.nonzero(as_tuple=False).view(-1),
            )
        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))

    def forward(self, value, mask=None, temporal_indices=None):
        batch, frames, height, width, channels = value.shape
        window_h, window_w = self.window_size
        head_channels = channels // self.n_head
        windows_h = math.ceil(height / window_h)
        windows_w = math.ceil(width / window_w)
        padded_h = windows_h * window_h
        padded_w = windows_w * window_w
        pad_right = padded_w - width
        pad_bottom = padded_h - height
        if pad_right or pad_bottom:
            value = F.pad(
                value,
                (0, 0, 0, pad_right, 0, pad_bottom, 0, 0),
                mode='constant',
                value=0,
            )
            mask = F.pad(
                mask,
                (0, 0, 0, pad_right, 0, pad_bottom, 0, 0),
                mode='constant',
                value=0,
            )

        query = self.query(value)
        key = self.key(value)
        val = self.value(value)
        window_count = windows_h * windows_w
        window_query = window_partition(
            query.contiguous(), self.window_size, self.n_head).view(
                batch, window_count, self.n_head, frames,
                window_h * window_w, head_channels)
        window_key = window_partition(
            key.contiguous(), self.window_size, self.n_head).view(
                batch, window_count, self.n_head, frames,
                window_h * window_w, head_channels)
        window_value = window_partition(
            val.contiguous(), self.window_size, self.n_head).view(
                batch, window_count, self.n_head, frames,
                window_h * window_w, head_channels)

        if any(size > 0 for size in self.expand_size):
            rolled = []
            for shifts in (
                (-self.expand_size[0], -self.expand_size[1]),
                (-self.expand_size[0], self.expand_size[1]),
                (self.expand_size[0], -self.expand_size[1]),
                (self.expand_size[0], self.expand_size[1]),
            ):
                rolled.append(torch.roll(key, shifts=shifts, dims=(2, 3)))
            rolled_key = torch.cat([
                window_partition(item, self.window_size, self.n_head).view(
                    batch, window_count, self.n_head, frames,
                    window_h * window_w, head_channels)
                for item in rolled
            ], dim=4)[:, :, :, :, self.valid_ind_rolled]
            rolled_value = []
            for shifts in (
                (-self.expand_size[0], -self.expand_size[1]),
                (-self.expand_size[0], self.expand_size[1]),
                (self.expand_size[0], -self.expand_size[1]),
                (self.expand_size[0], self.expand_size[1]),
            ):
                rolled_value.append(torch.roll(val, shifts=shifts, dims=(2, 3)))
            rolled_value = torch.cat([
                window_partition(item, self.window_size, self.n_head).view(
                    batch, window_count, self.n_head, frames,
                    window_h * window_w, head_channels)
                for item in rolled_value
            ], dim=4)[:, :, :, :, self.valid_ind_rolled]
            window_key = torch.cat((window_key, rolled_key), dim=4)
            window_value = torch.cat((window_value, rolled_value), dim=4)

        if self.pooling_token:
            pooled = self.pool_layer(
                value.view(batch * frames, padded_h, padded_w, channels)
                .permute(0, 3, 1, 2)
            )
            _, _, pooled_h, pooled_w = pooled.shape
            pooled = pooled.permute(0, 2, 3, 1).view(
                batch, frames, pooled_h, pooled_w, channels)
            pooled_key = self.key(pooled).unsqueeze(1).repeat(
                1, window_count, 1, 1, 1, 1)
            pooled_key = pooled_key.view(
                batch, window_count, frames, pooled_h, pooled_w,
                self.n_head, head_channels).permute(0, 1, 5, 2, 3, 4, 6)
            pooled_key = pooled_key.reshape(
                batch, window_count, self.n_head, frames,
                pooled_h * pooled_w, head_channels)
            pooled_value = self.value(pooled).unsqueeze(1).repeat(
                1, window_count, 1, 1, 1, 1)
            pooled_value = pooled_value.view(
                batch, window_count, frames, pooled_h, pooled_w,
                self.n_head, head_channels).permute(0, 1, 5, 2, 3, 4, 6)
            pooled_value = pooled_value.reshape(
                batch, window_count, self.n_head, frames,
                pooled_h * pooled_w, head_channels)
            window_key = torch.cat((window_key, pooled_key), dim=4)
            window_value = torch.cat((window_value, pooled_value), dim=4)

        output = torch.zeros_like(window_query)
        mask_windows = self.max_pool(mask.view(batch * mask.size(1), padded_h, padded_w))
        mask_windows = mask_windows.view(batch, mask.size(1), window_count)
        mask_windows = mask_windows.sum(dim=1)
        for batch_index in range(batch):
            masked_windows = mask_windows[batch_index].nonzero(
                as_tuple=False).flatten()
            if masked_windows.numel():
                query_masked = window_query[batch_index, masked_windows].view(
                    -1, self.n_head, frames * window_h * window_w,
                    head_channels)
                key_masked = window_key[batch_index, masked_windows]
                value_masked = window_value[batch_index, masked_windows]
                if temporal_indices is not None:
                    flat_indices = temporal_indices.reshape(-1)
                    key_masked = key_masked.index_select(2, flat_indices).reshape(
                        masked_windows.numel(), self.n_head, -1, head_channels)
                    value_masked = value_masked.index_select(2, flat_indices).reshape(
                        masked_windows.numel(), self.n_head, -1, head_channels)
                else:
                    key_masked = key_masked.reshape(
                        -1, self.n_head, frames * window_h * window_w,
                        head_channels)
                    value_masked = value_masked.reshape(
                        -1, self.n_head, frames * window_h * window_w,
                        head_channels)
                attention = torch.softmax(
                    query_masked @ key_masked.transpose(-2, -1)
                    / math.sqrt(head_channels), dim=-1)
                output[batch_index, masked_windows] = (
                    attention @ value_masked).view(
                        -1, self.n_head, frames, window_h * window_w,
                        head_channels)

            unmasked_windows = (mask_windows[batch_index] == 0).nonzero(
                as_tuple=False).flatten()
            query_unmasked = window_query[batch_index, unmasked_windows]
            key_unmasked = window_key[batch_index, unmasked_windows, :, :, :window_h * window_w]
            value_unmasked = window_value[batch_index, unmasked_windows, :, :, :window_h * window_w]
            attention = torch.softmax(
                query_unmasked @ key_unmasked.transpose(-2, -1)
                / math.sqrt(head_channels), dim=-1)
            output[batch_index, unmasked_windows] = attention @ value_unmasked

        output = output.view(
            batch, windows_h, windows_w, self.n_head, frames,
            window_h, window_w, head_channels)
        output = output.permute(
            0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(
                batch, frames, padded_h, padded_w, channels)
        if pad_right or pad_bottom:
            output = output[:, :, :height, :width]
        return self.proj_drop(self.proj(output))


class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, t2t_params):
        super().__init__()
        self.attention = SparseWindowAttention(
            dim, n_head, window_size, pool_size)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FusionFeedForward(dim, t2t_params=t2t_params)

    def forward(self, value, fold_size, mask, temporal_indices):
        batch, frames, height, width, channels = value.shape
        attention = self.attention(
            self.norm1(value), mask, temporal_indices)
        value = value + attention
        value = value + self.mlp(
            self.norm2(value).view(batch, frames * height * width, channels),
            fold_size,
        ).view(batch, frames, height, width, channels)
        return value


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths,
                 t2t_params):
        super().__init__()
        self.transformer = nn.Sequential(*[
            TemporalSparseTransformer(
                dim, n_head, window_size, pool_size, t2t_params)
            for _ in range(depths)
        ])
        self.depths = depths

    def forward(self, value, fold_size, mask, t_dilation=2):
        if self.depths % t_dilation:
            raise ValueError('Transformer depth must be divisible by t_dilation')
        frames = value.size(1)
        temporal_indices = [
            torch.arange(index, frames, t_dilation, device=value.device)
            for index in range(t_dilation)
        ] * (self.depths // t_dilation)
        for index, block in enumerate(self.transformer):
            value = block(value, fold_size, mask, temporal_indices[index])
        return value
