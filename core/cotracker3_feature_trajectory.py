import math

import torch
import torch.nn.functional as F

from core.cotracker3_ref_selector import CoTracker3RefSelector


def build_long_term_context_ids(center_frame, video_length, local_frame_ids,
                                window_size=60, temporal_stride=1):
    if video_length <= 0 or window_size <= 0 or temporal_stride <= 0:
        raise ValueError('video_length, window_size and temporal_stride must be positive')
    actual_window = min(int(window_size), int(video_length))
    start = int(center_frame) - actual_window // 2
    start = min(max(start, 0), video_length - actual_window)
    end = start + actual_window
    local_ids = sorted({int(value) for value in local_frame_ids
                        if start <= int(value) < end})
    target_length = max(
        len(local_ids), int(math.ceil(actual_window / temporal_stride)))
    if target_length >= actual_window:
        return list(range(start, end))
    local_set = set(local_ids)
    candidates = [value for value in range(start, end)
                  if value not in local_set]
    needed = min(target_length - len(local_ids), len(candidates))
    if needed:
        positions = torch.linspace(
            0, len(candidates) - 1, steps=needed).round().long()
        selected = {candidates[int(position)] for position in positions}
    else:
        selected = set()
    return sorted(local_ids + list(selected))


class CoTracker3FeatureTrajectoryExtractor:
    def __init__(self, checkpoint_path, repo_path, device='cuda',
                 context_window=60, query_batch_size=512,
                 max_queries=2048, max_support_queries=512,
                 support_interpolation_mode='nearest',
                 support_interpolation_k=4, iterations=2):
        if max_queries < 0 or max_support_queries < 0:
            raise ValueError('query limits must be non-negative')
        if support_interpolation_mode not in ('nearest', 'bilinear'):
            raise ValueError('unsupported support interpolation mode')
        self.device = torch.device(device)
        self.context_window = int(context_window)
        self.max_queries = int(max_queries)
        self.max_support_queries = int(max_support_queries)
        self.support_interpolation_mode = support_interpolation_mode
        self.support_interpolation_k = int(support_interpolation_k)
        self.backend = CoTracker3RefSelector(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            device=device,
            retrieval_window=context_window,
            query_batch_size=query_batch_size,
            iterations=iterations,
        )

    def _feature_cell_centers(self, feature_y, feature_x,
                              original_size, feature_size):
        original_h, original_w = original_size
        feature_h, feature_w = feature_size
        query_y = ((feature_y.float() + 0.5) * original_h / feature_h).long()
        query_x = ((feature_x.float() + 0.5) * original_w / feature_w).long()
        query_y = query_y.clamp(0, original_h - 1)
        query_x = query_x.clamp(0, original_w - 1)
        return query_x.float(), query_y.float()

    def _build_feature_queries(self, masks, local_context_indices, feature_size):
        if masks.shape[0] != 1:
            raise ValueError('feature trajectories require batch size 1')
        if local_context_indices.ndim == 2:
            local_context_indices = local_context_indices[0]
        local_context_indices = local_context_indices.long().to(masks.device)
        _, context_length, _, original_h, original_w = masks.shape
        feature_h, feature_w = feature_size
        downsampled_masks = F.interpolate(
            masks.reshape(context_length, 1, original_h, original_w).float(),
            size=(feature_h, feature_w),
            mode='nearest',
        ).reshape(1, context_length, 1, feature_h, feature_w) > 0.5

        queries = []
        local_indices = []
        feature_indices = []
        counts = []
        for local_index, context_index_tensor in enumerate(local_context_indices):
            context_index = int(context_index_tensor.item())
            feature_yx = torch.nonzero(
                downsampled_masks[0, context_index, 0], as_tuple=False)
            count = int(feature_yx.shape[0])
            counts.append(count)
            if count == 0:
                continue
            feature_y = feature_yx[:, 0]
            feature_x = feature_yx[:, 1]
            query_x, query_y = self._feature_cell_centers(
                feature_y, feature_x,
                (original_h, original_w), (feature_h, feature_w))
            times = torch.full_like(query_x, float(context_index))
            queries.append(torch.stack([times, query_x, query_y], dim=-1))
            local_indices.append(torch.full(
                (count,), local_index, dtype=torch.long, device=masks.device))
            feature_indices.append((feature_y * feature_w + feature_x).long())

        if not queries:
            empty_query = torch.empty(
                (0, 3), dtype=torch.float32, device=masks.device)
            empty_index = torch.empty(
                (0,), dtype=torch.long, device=masks.device)
            return empty_query, empty_index, empty_index, counts, counts, downsampled_masks

        queries = torch.cat(queries)
        local_indices = torch.cat(local_indices)
        feature_indices = torch.cat(feature_indices)
        candidate_counts = list(counts)
        if self.max_queries > 0 and queries.shape[0] > self.max_queries:
            keep = torch.linspace(
                0, queries.shape[0] - 1, steps=self.max_queries,
                device=queries.device).round().long()
            queries = queries[keep]
            local_indices = local_indices[keep]
            feature_indices = feature_indices[keep]
            counts = torch.bincount(
                local_indices, minlength=len(candidate_counts)).tolist()
        return (queries, local_indices, feature_indices, counts,
                candidate_counts, downsampled_masks)

    def _build_support_queries(self, target_queries, target_local_indices):
        if target_queries.shape[0] == 0:
            empty_index = torch.empty(
                (0,), dtype=torch.long, device=target_queries.device)
            return empty_index, target_queries.new_empty((0, 3)), empty_index, empty_index, []

        local_count = int(target_local_indices.max().item()) + 1
        by_local = [torch.nonzero(
            target_local_indices == index, as_tuple=False).flatten()
                    for index in range(local_count)]
        active = [index for index, values in enumerate(by_local)
                  if values.numel()]
        if self.max_support_queries <= 0:
            budgets = {index: int(by_local[index].numel()) for index in active}
        else:
            total = min(self.max_support_queries, int(target_queries.shape[0]))
            base, remainder = divmod(total, len(active))
            budgets = {
                index: min(
                    max(1, base + (position < remainder)),
                    int(by_local[index].numel()),
                )
                for position, index in enumerate(active)
            }

        selected_indices = []
        selected_local = []
        counts = []
        for local_index, values in enumerate(by_local):
            if values.numel() == 0:
                counts.append(0)
                continue
            budget = min(max(budgets[local_index], 1), int(values.numel()))
            if budget == values.numel():
                selected = values
            else:
                positions = torch.linspace(
                    0, values.numel() - 1, steps=budget,
                    device=values.device).round().long()
                selected = values[positions]
            selected_indices.append(selected)
            selected_local.append(torch.full_like(selected, local_index))
            counts.append(int(selected.numel()))

        selected_indices = torch.cat(selected_indices)
        selected_local = torch.cat(selected_local)
        return (
            torch.arange(
                target_queries.shape[0], dtype=torch.long,
                device=target_queries.device),
            target_queries[selected_indices],
            selected_indices,
            selected_local,
            counts,
        )

    @staticmethod
    def _interpolate_support_tracks(target_queries, target_local_indices,
                                    support_queries, support_local_indices,
                                    support_tracks, support_visibility,
                                    support_confidence, mode, interpolation_k):
        batch, context_length = support_tracks.shape[:2]
        target_count = int(target_queries.shape[0])
        tracks = support_tracks.new_zeros(
            batch, context_length, target_count, 2)
        visibility = support_visibility.new_zeros(
            batch, context_length, target_count)
        confidence = support_confidence.new_zeros(
            batch, context_length, target_count)

        for local_index in torch.unique(target_local_indices).tolist():
            target_ids = torch.nonzero(
                target_local_indices == int(local_index), as_tuple=False).flatten()
            support_ids = torch.nonzero(
                support_local_indices == int(local_index), as_tuple=False).flatten()
            if target_ids.numel() == 0 or support_ids.numel() == 0:
                continue
            target_xy = target_queries[target_ids, 1:].float()
            support_xy = support_queries[support_ids, 1:].float()
            distances = torch.cdist(target_xy, support_xy)
            if mode == 'nearest':
                nearest = distances.argmin(dim=-1).unsqueeze(-1)
                weights = torch.ones(
                    target_ids.numel(), 1, device=distances.device,
                    dtype=distances.dtype)
            else:
                k = min(interpolation_k, int(support_ids.numel()))
                nearest_distances, nearest = torch.topk(
                    distances, k=k, dim=-1, largest=False)
                exact = nearest_distances <= 1e-6
                weights = 1.0 / nearest_distances.clamp_min(1e-4)
                if exact.any():
                    exact_weights = torch.zeros_like(weights)
                    exact_weights[exact] = 1.0
                    weights = torch.where(
                        exact.any(dim=-1, keepdim=True), exact_weights, weights)
                weights = weights / weights.sum(
                    dim=-1, keepdim=True).clamp_min(1e-6)

            selected = support_ids[nearest]
            selected_xy = support_queries[selected, 1:].float()
            selected_tracks = torch.nan_to_num(
                support_tracks[:, :, selected], nan=0.0, posinf=0.0, neginf=0.0)
            displacement = selected_tracks - selected_xy[None, None]
            tracks[:, :, target_ids] = target_xy[None, None] + (
                displacement * weights[None, None, :, :, None]).sum(dim=3)
            visibility[:, :, target_ids] = (
                support_visibility[:, :, selected] * weights[None, None]).sum(dim=3)
            confidence[:, :, target_ids] = (
                support_confidence[:, :, selected] * weights[None, None]).sum(dim=3)
        return tracks, visibility, confidence

    @torch.no_grad()
    def compute(self, clean_context_frames, context_masks,
                local_context_indices, feature_size=None):
        if clean_context_frames.ndim != 5 or clean_context_frames.shape[0] != 1:
            raise ValueError('clean_context_frames must have shape [1,T,3,H,W]')
        if context_masks.shape[:2] != clean_context_frames.shape[:2]:
            raise ValueError('context_masks must match context frames')

        original_h, original_w = clean_context_frames.shape[-2:]
        if feature_size is None:
            feature_size = (original_h // 4, original_w // 4)
        (queries, query_local_indices, query_feature_indices, query_counts,
         candidate_counts, downsampled_masks) = self._build_feature_queries(
             context_masks, local_context_indices, feature_size)
        (_, support_queries, _, support_local_indices, support_counts) = (
            self._build_support_queries(queries, query_local_indices))

        if support_queries.shape[0] == 0:
            context_length = clean_context_frames.shape[1]
            tracks = clean_context_frames.new_empty(
                (1, context_length, queries.shape[0], 2))
            visibility = clean_context_frames.new_zeros(
                (1, context_length, queries.shape[0]))
            confidence = visibility.clone()
        else:
            video = ((clean_context_frames.float() + 1.0) * 127.5).clamp(0, 255)
            support_tracks, support_visibility, support_confidence = (
                self.backend._run_tracking(video.to(self.device), support_queries))
            tracks, visibility, confidence = self._interpolate_support_tracks(
                queries,
                query_local_indices,
                support_queries,
                support_local_indices,
                support_tracks,
                support_visibility,
                support_confidence,
                self.support_interpolation_mode,
                self.support_interpolation_k,
            )

        return {
            'format': 'cotracker3_sparse_feature',
            'trajectory': tracks,
            'visibility': visibility,
            'confidence': confidence,
            'queries': queries.unsqueeze(0),
            'query_local_indices': query_local_indices.unsqueeze(0),
            'query_feature_indices': query_feature_indices.unsqueeze(0),
            'query_counts_per_local': query_counts,
            'candidate_query_counts_per_local': candidate_counts,
            'num_candidate_queries': int(sum(candidate_counts)),
            'num_queries': int(queries.shape[0]),
            'num_tracking_queries': int(support_queries.shape[0]),
            'cotracker_iterations': int(self.backend.iterations),
            'local_context_indices': local_context_indices,
            'context_masks_feature': downsampled_masks,
            'original_size': (int(original_h), int(original_w)),
            'feature_size': tuple(int(value) for value in feature_size),
            'trajectory_source': 'mask_internal_support_interpolation',
            'support_queries': support_queries.unsqueeze(0),
            'support_query_local_indices': support_local_indices.unsqueeze(0),
            'support_query_counts_per_local': support_counts,
            'support_interpolation_mode': self.support_interpolation_mode,
            'support_interpolation_k': int(self.support_interpolation_k),
        }
