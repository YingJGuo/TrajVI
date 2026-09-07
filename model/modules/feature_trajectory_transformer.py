import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTrajectoryTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f'dim={dim} must be divisible by n_head={n_head}')

        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward_local_kv(
            self,
            query_sequence,
            source_sequence,
            source_valid,
            target_update):
        batch_size, num_tracks, num_frames, channels = query_sequence.shape
        patch_tokens = source_sequence.shape[3]
        query_flat = query_sequence.reshape(
            batch_size * num_tracks, num_frames, channels)
        source_flat = source_sequence.reshape(
            batch_size * num_tracks, num_frames * patch_tokens, channels)
        source_valid_flat = source_valid.reshape(
            batch_size * num_tracks, num_frames * patch_tokens)
        target_flat = target_update.reshape(batch_size * num_tracks, num_frames)

        query_norm = self.norm1(query_flat)
        source_norm = self.norm1(source_flat)
        query = self.query(query_norm)
        key = self.key(source_norm)
        value = self.value(source_norm)

        query = query.view(
            -1, num_frames, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(
            -1, num_frames * patch_tokens, self.n_head,
            self.head_dim).transpose(1, 2)
        value = value.view(
            -1, num_frames * patch_tokens, self.n_head,
            self.head_dim).transpose(1, 2)

        has_source = source_valid_flat.any(dim=-1)
        safe_source = source_valid_flat.clone()
        safe_source[~has_source, 0] = True

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~safe_source[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        attn_output = weights @ value
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            -1, num_frames, channels)
        attn_output = self.proj_drop(self.proj(attn_output))

        update_mask = target_flat & has_source[:, None]
        update_weight = update_mask.unsqueeze(-1).to(query_flat.dtype)
        query_flat = query_flat + attn_output * update_weight
        query_flat = query_flat + self.ffn(self.norm2(query_flat)) * update_weight
        return query_flat.view(batch_size, num_tracks, num_frames, channels)


class FeatureTrajectoryTransformer(nn.Module):
    def __init__(self, dim=128, n_head=4, depth=2, mlp_ratio=2.0,
                 dropout=0.0, conf_threshold=0.3, output_init_scale=0.1):
        super().__init__()
        self.dim = dim
        self.conf_threshold = conf_threshold
        self.output_init_scale = float(output_init_scale)
        self.blocks = nn.ModuleList([
            FeatureTrajectoryTransformerBlock(
                dim=dim,
                n_head=n_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            for block in self.blocks:
                block.proj.weight.mul_(self.output_init_scale)
                block.ffn[3].weight.mul_(self.output_init_scale)

    @staticmethod
    def _with_batch(tensor, batch_size, name, expected_ndim):
        if tensor.ndim == expected_ndim - 1:
            if batch_size != 1:
                raise ValueError(
                    f"trajectory_result['{name}'] has no batch dimension for batch size {batch_size}"
                )
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != expected_ndim or tensor.shape[0] != batch_size:
            raise ValueError(
                f"trajectory_result['{name}'] must have batch size {batch_size}; got {tuple(tensor.shape)}"
            )
        return tensor

    @staticmethod
    def _normalize_local_context_indices(local_context_indices, batch_size, device):
        if local_context_indices.ndim == 1:
            local_context_indices = local_context_indices.unsqueeze(0)
        if local_context_indices.ndim != 2:
            raise ValueError(
                'local_context_indices must have shape [L] or [B,L]; '
                f'got {tuple(local_context_indices.shape)}')
        if local_context_indices.shape[0] == 1 and batch_size > 1:
            local_context_indices = local_context_indices.expand(batch_size, -1)
        if local_context_indices.shape[0] != batch_size:
            raise ValueError('local_context_indices batch size does not match context features')
        return local_context_indices.long().to(device)

    @staticmethod
    def _scatter_average_selected(
            delta, spatial_idx, target_update, local_context_indices, feature_size):
        batch_size, num_tracks, _, channels = delta.shape
        feature_h, feature_w = feature_size
        num_positions = feature_h * feature_w
        num_local = local_context_indices.shape[1]

        delta_bt = delta.permute(0, 2, 1, 3).contiguous()
        target_bt = target_update.permute(0, 2, 1).contiguous()
        gather_tracks = local_context_indices[:, :, None].expand(
            batch_size, num_local, num_tracks)
        gather_delta = gather_tracks.unsqueeze(-1).expand(
            batch_size, num_local, num_tracks, channels)
        local_delta = torch.gather(delta_bt, 1, gather_delta)
        local_target = torch.gather(target_bt, 1, gather_tracks)
        local_spatial = torch.gather(spatial_idx, 1, gather_tracks)

        output_sum = delta.new_zeros(
            batch_size, num_local, num_positions, channels)
        output_sum.scatter_add_(
            2,
            local_spatial.unsqueeze(-1).expand(-1, -1, -1, channels),
            local_delta * local_target.unsqueeze(-1).to(delta.dtype),
        )
        count = delta.new_zeros(batch_size, num_local, num_positions, 1)
        count.scatter_add_(
            2, local_spatial.unsqueeze(-1), local_target.unsqueeze(-1).to(delta.dtype))
        output = output_sum / count.clamp_min(1)
        output = output.view(batch_size, num_local, feature_h, feature_w, channels)
        output = output.permute(0, 1, 4, 2, 3).contiguous()
        count = count.view(batch_size, num_local, feature_h, feature_w, 1)
        return output, count

    def forward_sparse(
            self,
            context_feat,
            trajectory,
            visibility,
            confidence,
            context_masks,
            local_context_indices,
            original_size,
            conf_threshold=None):
        batch_size, context_length, channels, feature_h, feature_w = context_feat.shape
        if channels != self.dim:
            raise ValueError(f'Expected feature dim {self.dim}, got {channels}')
        if context_masks.shape != (
                batch_size, context_length, 1, feature_h, feature_w):
            raise ValueError('context_masks shape does not match context features')
        local_context_indices = self._normalize_local_context_indices(
            local_context_indices, batch_size, context_feat.device)
        num_local = local_context_indices.shape[1]

        trajectory = self._with_batch(
            trajectory, batch_size, 'trajectory', 4).to(context_feat.device)
        visibility = self._with_batch(
            visibility, batch_size, 'visibility', 3).to(context_feat.device)
        confidence = self._with_batch(
            confidence, batch_size, 'confidence', 3).to(context_feat.device)
        if trajectory.shape[1] != context_length:
            raise ValueError('Sparse trajectory T does not match context feature T')
        num_tracks = trajectory.shape[2]
        if visibility.shape != (batch_size, context_length, num_tracks):
            raise ValueError('Sparse visibility shape does not match trajectory')
        if confidence.shape != (batch_size, context_length, num_tracks):
            raise ValueError('Sparse confidence shape does not match trajectory')

        if num_tracks == 0:
            return context_feat.new_zeros(
                batch_size, num_local, channels, feature_h, feature_w)

        original_h, original_w = original_size
        feature_x = trajectory[..., 0].to(context_feat.dtype) * (
            float(feature_w) / float(original_w))
        feature_y = trajectory[..., 1].to(context_feat.dtype) * (
            float(feature_h) / float(original_h))
        finite = torch.isfinite(feature_x) & torch.isfinite(feature_y)
        in_bounds = finite & (feature_x >= 0) & (feature_x < feature_w)
        in_bounds &= (feature_y >= 0) & (feature_y < feature_h)
        safe_x = torch.nan_to_num(feature_x, nan=0.0, posinf=0.0, neginf=0.0)
        safe_y = torch.nan_to_num(feature_y, nan=0.0, posinf=0.0, neginf=0.0)
        safe_x = safe_x.long().clamp(0, feature_w - 1)
        safe_y = safe_y.long().clamp(0, feature_h - 1)
        spatial_idx = safe_y * feature_w + safe_x

        flat_mask = context_masks[:, :, 0].reshape(
            batch_size, context_length, feature_h * feature_w)
        mask_on_track = torch.gather(flat_mask, 2, spatial_idx)
        threshold = self.conf_threshold if conf_threshold is None else conf_threshold
        point_valid = (visibility * confidence > threshold) & in_bounds
        source_time_valid = point_valid & (mask_on_track < 0.5)

        update_frames = torch.zeros(
            batch_size, context_length, dtype=torch.bool, device=context_feat.device)
        update_frames.scatter_(1, local_context_indices, True)
        target_update = point_valid & (mask_on_track >= 0.5)
        target_update &= update_frames[:, :, None]

        radius = 0
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=context_feat.device),
            torch.arange(-radius, radius + 1, device=context_feat.device),
            indexing='ij',
        )
        offsets_y = offsets_y.reshape(1, 1, 1, -1)
        offsets_x = offsets_x.reshape(1, 1, 1, -1)
        patch_x = safe_x.unsqueeze(-1) + offsets_x
        patch_y = safe_y.unsqueeze(-1) + offsets_y
        patch_in_bounds = in_bounds.unsqueeze(-1)
        patch_in_bounds = patch_in_bounds & (
            (patch_x >= 0) & (patch_x < feature_w)
            & (patch_y >= 0) & (patch_y < feature_h)
        )
        safe_patch_x = patch_x.clamp(0, feature_w - 1)
        safe_patch_y = patch_y.clamp(0, feature_h - 1)
        patch_spatial_idx = safe_patch_y * feature_w + safe_patch_x
        patch_mask = torch.gather(
            flat_mask,
            2,
            patch_spatial_idx.reshape(batch_size, context_length, -1),
        ).reshape_as(patch_spatial_idx)
        source_valid = source_time_valid.unsqueeze(-1) & patch_in_bounds
        source_valid = source_valid & (patch_mask < 0.5)

        track_keep = source_valid.any(dim=(1, 3)) & target_update.any(dim=1)
        keep_any_batch = track_keep.any(dim=0)
        if not keep_any_batch.any():
            return context_feat.new_zeros(
                batch_size, num_local, channels, feature_h, feature_w)

        spatial_idx = spatial_idx[:, :, keep_any_batch]
        patch_spatial_idx = patch_spatial_idx[:, :, keep_any_batch]
        source_valid = source_valid[:, :, keep_any_batch]
        target_update = target_update[:, :, keep_any_batch]
        track_keep = track_keep[:, keep_any_batch]
        source_valid &= track_keep[:, None, :, None]
        target_update &= track_keep[:, None, :]

        flat_feat = context_feat.permute(0, 1, 3, 4, 2).reshape(
            batch_size, context_length, feature_h * feature_w, channels)
        sequence = torch.gather(
            flat_feat,
            2,
            spatial_idx.unsqueeze(-1).expand(-1, -1, -1, channels),
        ).permute(0, 2, 1, 3).contiguous()
        patch_tokens = patch_spatial_idx.shape[-1]
        source_sequence = torch.gather(
            flat_feat,
            2,
            patch_spatial_idx.reshape(
                batch_size, context_length, -1).unsqueeze(-1).expand(
                    -1, -1, -1, channels),
        ).reshape(
            batch_size, context_length, -1, patch_tokens, channels
        ).permute(0, 2, 1, 3, 4).contiguous()
        original_sequence = sequence
        source_valid = source_valid.permute(0, 2, 1, 3).contiguous()
        target_update = target_update.permute(0, 2, 1).contiguous()

        for block in self.blocks:
            sequence = block.forward_local_kv(
                sequence,
                source_sequence,
                source_valid,
                target_update,
            )

        delta = (sequence - original_sequence) * target_update.unsqueeze(-1).to(sequence.dtype)
        feature_delta, _ = self._scatter_average_selected(
            delta,
            spatial_idx,
            target_update,
            local_context_indices,
            (feature_h, feature_w),
        )
        return feature_delta
