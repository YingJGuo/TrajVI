import torch
import torch.nn as nn
import torch.nn.functional as F

from model.trajectory_propagation import TrajectoryDeformableAlignment


def _warp_feature(source, offset):
    _, _, height, width = source.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing='ij',
    )
    sample_x = x[None] + offset[:, 0]
    sample_y = y[None] + offset[:, 1]
    grid = torch.stack([
        2.0 * sample_x / max(width - 1, 1) - 1.0,
        2.0 * sample_y / max(height - 1, 1) - 1.0,
    ], dim=-1)
    return F.grid_sample(
        source, grid, mode='bilinear', padding_mode='zeros',
        align_corners=True)


def _fit_local_knn_region_offset(target_xy, source_xy, weights, height, width,
                                 roi, neighbors=4, sigma=3.0):
    dtype = target_xy.dtype
    device = target_xy.device
    dense = torch.zeros(height, width, 2, device=device, dtype=dtype)
    roi_yx = torch.nonzero(roi, as_tuple=False)
    if target_xy.numel() == 0 or roi_yx.numel() == 0:
        return dense.permute(2, 0, 1).contiguous()

    roi_xy = roi_yx[:, [1, 0]].to(dtype)
    k = min(max(int(neighbors), 1), int(target_xy.shape[0]))
    distances = torch.cdist(roi_xy, target_xy)
    nearest_distance, nearest_index = distances.topk(
        k, dim=1, largest=False)
    displacement = source_xy - target_xy
    local_displacement = displacement[nearest_index]
    local_confidence = weights.float().clamp_min(1e-3)[nearest_index]
    local_median = local_displacement.median(dim=1, keepdim=True).values
    residual = torch.linalg.vector_norm(
        local_displacement - local_median, dim=-1)
    robust_weight = 1.0 / (1.0 + residual.square() / 4.0)
    spatial_weight = torch.exp(
        -nearest_distance.square() / (2.0 * max(float(sigma), 1e-3) ** 2))
    combined_weight = local_confidence * robust_weight * spatial_weight
    interpolated = (
        (local_displacement * combined_weight.unsqueeze(-1)).sum(dim=1)
        / combined_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    )
    dense[roi_yx[:, 0], roi_yx[:, 1]] = interpolated.to(dtype)
    return dense.permute(2, 0, 1).contiguous()


class RegionWarpTrajectoryDCN(nn.Module):
    def __init__(self, channels=128, top_sources=4, conf_threshold=0.3,
                 max_residue_magnitude=1.5, roi_radius=2, geometry_knn=4,
                 geometry_sigma=3.0, residual_scale=0.25,
                 fixed_update_gate=0.25):
        super().__init__()
        if top_sources <= 0:
            raise ValueError('top_sources must be positive')
        if roi_radius < 0 or geometry_knn <= 0:
            raise ValueError('roi_radius and geometry_knn must be non-negative')
        if not 0.0 <= fixed_update_gate <= 1.0:
            raise ValueError('fixed_update_gate must be in [0, 1]')
        if residual_scale < 0.0:
            raise ValueError('residual_scale must be non-negative')

        self.channels = int(channels)
        self.top_sources = int(top_sources)
        self.conf_threshold = float(conf_threshold)
        self.roi_radius = int(roi_radius)
        self.geometry_knn = int(geometry_knn)
        self.geometry_sigma = float(geometry_sigma)
        self.residual_scale = float(residual_scale)
        self.fixed_update_gate = float(fixed_update_gate)

        self.align = TrajectoryDeformableAlignment(
            channels, channels, 3, padding=1, deform_groups=16,
            max_residue_magnitude=max_residue_magnitude,
        )
        selector_hidden = max(channels // 2, 16)
        self.source_selector = nn.Sequential(
            nn.Conv2d(2 * channels + 4, selector_hidden, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(selector_hidden, 1, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels + 2, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.align.weight.zero_()
            channel_index = torch.arange(self.channels)
            self.align.weight[channel_index, channel_index, 1, 1] = 1.0
            if self.align.bias is not None:
                self.align.bias.zero_()
            last_offset = self.align.conv_offset[-1]
            last_offset.weight.zero_()
            last_offset.bias.zero_()
            last_offset.bias[2 * last_offset.bias.numel() // 3:] = 5.0
            self.fuse[-1].weight.zero_()
            self.fuse[-1].bias.zero_()
            self.source_selector[-1].weight.zero_()
            self.source_selector[-1].bias.zero_()

    @torch.no_grad()
    def _build_routing(self, local_feat, context_masks,
                       local_context_indices, trajectory_result):
        batch, local_length, _, height, width = local_feat.shape
        if batch != 1:
            raise ValueError('trajectory DCN requires batch size 1')

        trajectory = trajectory_result['trajectory']
        visibility = trajectory_result['visibility']
        confidence = trajectory_result['confidence']
        query_local = trajectory_result['query_local_indices']
        query_feature = trajectory_result['query_feature_indices']
        if trajectory.ndim == 3:
            trajectory = trajectory.unsqueeze(0)
        if visibility.ndim == 2:
            visibility = visibility.unsqueeze(0)
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(0)
        if query_local.ndim == 1:
            query_local = query_local.unsqueeze(0)
        if query_feature.ndim == 1:
            query_feature = query_feature.unsqueeze(0)

        device = local_feat.device
        trajectory = trajectory[0].to(device).float()
        visibility = visibility[0].to(device).float()
        confidence = confidence[0].to(device).float()
        query_local = query_local[0].to(device).long()
        query_feature = query_feature[0].to(device).long()
        local_context_indices = local_context_indices[0].to(device).long()
        context_masks = context_masks[0, :, 0]
        context_length, num_tracks = trajectory.shape[:2]
        if query_local.numel() != num_tracks:
            raise ValueError('trajectory tracks and sparse query mapping differ')
        if context_masks.shape != (context_length, height, width):
            raise ValueError('context masks do not match trajectory feature grid')
        original_h, original_w = trajectory_result['original_size']

        lattice_x = trajectory[..., 0] * width / original_w - 0.5
        lattice_y = trajectory[..., 1] * height / original_h - 0.5
        cell_x = torch.floor(trajectory[..., 0] * width / original_w).long()
        cell_y = torch.floor(trajectory[..., 1] * height / original_h).long()
        finite = torch.isfinite(lattice_x) & torch.isfinite(lattice_y)
        in_bounds = finite & (cell_x >= 0) & (cell_x < width)
        in_bounds &= (cell_y >= 0) & (cell_y < height)
        safe_x = cell_x.clamp(0, width - 1)
        safe_y = cell_y.clamp(0, height - 1)
        time_index = torch.arange(context_length, device=device)[:, None]
        mask_on_track = context_masks[time_index, safe_y, safe_x]
        reliability = visibility * confidence
        reliable = (reliability > self.conf_threshold) & in_bounds
        ranking_valid = reliable & (mask_on_track < 0.5)

        source_indices = torch.zeros(
            local_length, self.top_sources, dtype=torch.long, device=device)
        offsets = local_feat.new_zeros(
            local_length, self.top_sources, 2, height, width)
        target_rois = torch.zeros(
            local_length, 1, height, width, dtype=torch.bool, device=device)
        nonlocal_mask = torch.ones(context_length, dtype=torch.bool, device=device)
        nonlocal_mask[local_context_indices] = False
        nonlocal_count = int(nonlocal_mask.sum())
        if nonlocal_count == 0:
            return source_indices, offsets, target_rois
        selected_count = min(self.top_sources, nonlocal_count)

        for local_index in range(local_length):
            tracks = torch.nonzero(
                query_local == local_index, as_tuple=False).flatten()
            if tracks.numel() == 0:
                continue
            target_flat = query_feature[tracks]
            target_y = torch.div(target_flat, width, rounding_mode='floor')
            target_x = target_flat.remainder(width)
            target_mask = context_masks[local_context_indices[local_index]]
            roi_seed = torch.zeros_like(target_mask, dtype=torch.bool)
            roi_seed[target_y, target_x] = True
            roi = F.max_pool2d(
                roi_seed[None, None].float(),
                2 * self.roi_radius + 1, 1, self.roi_radius,
            )[0, 0] > 0.5
            roi = roi & (target_mask > 0.5)
            target_rois[local_index, 0] = roi

            counts = ranking_valid[:, tracks].sum(dim=1)
            counts = counts.masked_fill(~nonlocal_mask, -1)
            chosen = counts.topk(selected_count).indices
            source_indices[local_index, :selected_count] = chosen
            if selected_count < self.top_sources:
                source_indices[local_index, selected_count:] = chosen[0]

            for slot, source_t in enumerate(chosen):
                geometry_valid = reliable[source_t, tracks]
                if not bool(geometry_valid.any()):
                    continue
                target_xy = torch.stack([
                    target_x[geometry_valid].float(),
                    target_y[geometry_valid].float(),
                ], dim=-1)
                source_xy = torch.stack([
                    lattice_x[source_t, tracks[geometry_valid]],
                    lattice_y[source_t, tracks[geometry_valid]],
                ], dim=-1)
                offsets[local_index, slot] = _fit_local_knn_region_offset(
                    target_xy,
                    source_xy,
                    reliability[source_t, tracks[geometry_valid]],
                    height,
                    width,
                    roi,
                    neighbors=self.geometry_knn,
                    sigma=self.geometry_sigma,
                ).to(local_feat.dtype)

        return source_indices, offsets, target_rois

    def forward(self, local_feat, context_feat, context_masks,
                local_context_indices, trajectory_result):
        if trajectory_result.get('format') != 'cotracker3_sparse_feature':
            raise ValueError('unsupported trajectory format')
        if local_context_indices.ndim == 1:
            local_context_indices = local_context_indices.unsqueeze(0)
        source_indices, offsets, target_rois = self._build_routing(
            local_feat, context_masks, local_context_indices, trajectory_result)
        _, local_length, channels, height, width = local_feat.shape
        outputs = []

        for local_index in range(local_length):
            source_ids = source_indices[local_index]
            source = context_feat[:, source_ids].reshape(
                -1, channels, height, width)
            target = local_feat[:, local_index].expand(
                self.top_sources, -1, -1, -1)
            dense_offset = offsets[local_index]
            source_mask = context_masks[:, source_ids].reshape(
                -1, 1, height, width).to(local_feat.dtype)
            region = target_rois[local_index:local_index + 1].expand(
                self.top_sources, -1, -1, -1).to(local_feat.dtype)
            warped_raw = _warp_feature(source, dense_offset)
            warped_mask_raw = _warp_feature(
                source_mask, dense_offset).clamp(0, 1)
            support_raw = _warp_feature(
                torch.ones_like(source_mask), dense_offset).clamp(0, 1)
            warped = warped_raw * region + source * (1.0 - region)
            warped_mask = warped_mask_raw * region + source_mask * (1.0 - region)
            support = support_raw * region + (1.0 - region)
            source_valid = ((1.0 - warped_mask) * support).clamp(0, 1) * region
            condition = torch.cat([
                target,
                warped,
                dense_offset,
                source_valid,
                torch.ones_like(source_valid),
                context_masks[:, local_context_indices[0, local_index]].expand(
                    self.top_sources, -1, -1, -1).to(local_feat.dtype),
            ], dim=1)
            aligned = self.align(source, condition, dense_offset)
            aligned = aligned * region + source * (1.0 - region)

            current = local_feat[:, local_index]
            target_mask = context_masks[
                :, local_context_indices[0, local_index]].to(local_feat.dtype)
            aligned_sources = aligned.reshape(
                1, self.top_sources, channels, height, width)
            valid_sources = source_valid.reshape(
                1, self.top_sources, 1, height, width)
            target_sources = current[:, None].expand(
                -1, self.top_sources, -1, -1, -1)
            offset_magnitude = torch.linalg.vector_norm(
                dense_offset, dim=1, keepdim=True).reshape(
                    1, self.top_sources, 1, height, width)
            feature_difference = (
                aligned_sources - target_sources).abs().mean(
                    dim=2, keepdim=True)
            target_mask_sources = target_mask[:, None].expand(
                -1, self.top_sources, -1, -1, -1)
            selector_input = torch.cat([
                target_sources,
                aligned_sources,
                valid_sources,
                target_mask_sources,
                offset_magnitude,
                feature_difference,
            ], dim=2).reshape(
                self.top_sources, 2 * channels + 4, height, width)
            source_logits = self.source_selector(selector_input).reshape(
                1, self.top_sources, 1, height, width)
            source_logits = source_logits + torch.log(
                0.05 + 0.95 * valid_sources)
            source_weight = torch.softmax(source_logits, dim=1)
            aligned_mean = (aligned_sources * source_weight).sum(dim=1)
            valid_mean = (valid_sources * source_weight).sum(dim=1)
            fusion_input = torch.cat([
                current, aligned_mean, valid_mean, target_mask,
            ], dim=1)
            delta = self.fuse(fusion_input)
            outputs.append(
                current + self.residual_scale * delta
                * target_rois[local_index:local_index + 1].to(delta.dtype)
                * self.fixed_update_gate)

        return torch.stack(outputs, dim=1)
