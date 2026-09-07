import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from einops import rearrange

from model.modules.base_module import BaseNetwork
from model.modules.deformconv import ModulatedDeformConv2d
from model.modules.feature_trajectory_transformer import FeatureTrajectoryTransformer
from model.flow_utils import flow_warp
from model.modules.longterm_trajectory_dcn import RegionWarpTrajectoryDCN
from model.modules.sparse_transformer import (
    SoftComp,
    SoftSplit,
    TemporalSparseTransformerBlock,
)
from .misc import constant_init


def _length_sq(value):
    return torch.sum(torch.square(value), dim=1, keepdim=True)


def _flow_consistency(flow_forward, flow_backward,
                      alpha1=0.01, alpha2=0.5):
    backward_warped = flow_warp(
        flow_backward, flow_forward.permute(0, 2, 3, 1))
    difference = flow_forward + backward_warped
    magnitude = _length_sq(flow_forward) + _length_sq(backward_warped)
    threshold = alpha1 * magnitude + alpha2
    return (_length_sq(difference) < threshold).to(flow_forward)


class DeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)
        super().__init__(*args, **kwargs)
        input_channels = 2 * self.out_channels + 5
        self.conv_offset = nn.Sequential(
            nn.Conv2d(input_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, feature, condition, flow):
        output = self.conv_offset(condition)
        offset_x, offset_y, mask = torch.chunk(output, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((offset_x, offset_y), dim=1))
        offset = offset + flow.flip(1).repeat(
            1, offset.size(1) // 2, 1, 1)
        return torchvision.ops.deform_conv2d(
            feature,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            torch.sigmoid(mask),
        )


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ('backward_1', 'forward_1')
        for module_name in self.prop_list:
            self.deform_align[module_name] = DeformableAlignment(
                channel, channel, 3, padding=1, deform_groups=16)
            self.backbone[module_name] = nn.Sequential(
                nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
        )

    def forward(self, feature, flows_forward, flows_backward, mask,
                interpolation='bilinear'):
        batch, frames, channels, height, width = feature.shape
        features = {'input': [feature[:, i] for i in range(frames)]}
        masks = {'input': [mask[:, i] for i in range(frames)]}

        for propagation_index, module_name in enumerate(self.prop_list):
            features[module_name] = []
            masks[module_name] = []
            if module_name.startswith('backward'):
                frame_ids = list(range(frames - 1, -1, -1))
                flow_ids = frame_ids
                propagation_flows = flows_forward
                consistency_flows = flows_backward
            else:
                frame_ids = list(range(frames))
                flow_ids = range(-1, frames - 1)
                propagation_flows = flows_backward
                consistency_flows = flows_forward

            previous_name = (
                'input' if propagation_index == 0
                else self.prop_list[propagation_index - 1]
            )
            for step, frame_id in enumerate(frame_ids):
                current_feature = features[previous_name][frame_id]
                current_mask = masks[previous_name][frame_id]
                if step == 0:
                    propagated_feature = current_feature
                else:
                    flow = propagation_flows[:, flow_ids[step]]
                    check_flow = consistency_flows[:, flow_ids[step]]
                    valid = _flow_consistency(flow, check_flow)
                    warped = flow_warp(
                        propagated_feature,
                        flow.permute(0, 2, 3, 1),
                        interpolation,
                    )
                    condition = torch.cat(
                        [current_feature, warped, flow, valid, current_mask],
                        dim=1,
                    )
                    propagated_feature = self.deform_align[module_name](
                        propagated_feature, condition, flow)

                refinement_input = torch.cat(
                    [current_feature, propagated_feature, current_mask], dim=1)
                propagated_feature = propagated_feature + self.backbone[
                    module_name](refinement_input)
                features[module_name].append(propagated_feature)
                masks[module_name].append(current_mask)

            if module_name.startswith('backward'):
                features[module_name].reverse()
                masks[module_name].reverse()

        output_backward = torch.stack(features['backward_1'], dim=1)
        output_forward = torch.stack(features['forward_1'], dim=1)
        output_backward_flat = output_backward.reshape(-1, channels, height, width)
        output_forward_flat = output_forward.reshape(-1, channels, height, width)

        mask_input = mask.reshape(-1, 2, height, width)
        output = self.fuse(torch.cat(
            [output_backward_flat, output_forward_flat, mask_input], dim=1))
        output = output + feature.reshape(-1, channels, height, width)
        output_mask = None

        return (
            output_backward,
            output_forward,
            output.reshape(batch, -1, channels, height, width),
            output_mask,
        )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.group = (1, 2, 4, 8, 1)
        self.layers = nn.ModuleList([
            nn.Conv2d(5, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, 3, 1, 1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, 3, 1, 1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, 3, 1, 1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ])

    def forward(self, value):
        batch_time = value.size(0)
        output = value
        for index, layer in enumerate(self.layers):
            if index == 8:
                skip = output
                _, _, height, width = skip.shape
            if index > 8 and index % 2 == 0:
                groups = self.group[(index - 8) // 2]
                skip_grouped = skip.view(batch_time, groups, -1, height, width)
                output_grouped = output.view(batch_time, groups, -1, height, width)
                output = torch.cat(
                    [skip_grouped, output_grouped], dim=2).view(
                        batch_time, -1, height, width)
            output = layer(output)
        return output


class Deconv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, 1, padding)

    def forward(self, value):
        return self.conv(F.interpolate(
            value, scale_factor=2, mode='bilinear', align_corners=True))


class InpaintGenerator(BaseNetwork):
    def __init__(self, model_path=None):
        super().__init__()
        channel = 128
        hidden = 512
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
        }

        self.encoder = Encoder()
        self.decoder = nn.Sequential(
            Deconv(channel, 128, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            Deconv(64, 64, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        self.feat_prop_module = BidirectionalPropagation(128)
        self.transformers = TemporalSparseTransformerBlock(
            dim=hidden,
            n_head=4,
            window_size=(5, 9),
            pool_size=(4, 4),
            depths=8,
            t2t_params=t2t_params,
        )
        self.feature_trajectory_transformer = FeatureTrajectoryTransformer(
            dim=channel,
            n_head=4,
            depth=2,
            mlp_ratio=2.0,
            dropout=0.0,
            conf_threshold=0.3,
            output_init_scale=0.1,
        )
        self.longterm_trajectory_dcn = RegionWarpTrajectoryDCN(
            channels=channel,
            top_sources=4,
            conf_threshold=0.3,
            max_residue_magnitude=1.5,
            roi_radius=2,
            geometry_knn=4,
            geometry_sigma=3.0,
            residual_scale=0.25,
            fixed_update_gate=0.25,
        )

        self.init_weights()
        self.feature_trajectory_transformer.reset_parameters()
        self.longterm_trajectory_dcn.reset_parameters()

        if model_path is not None:
            self._load_checkpoint(model_path)
        self.print_network()

    def _normalize_checkpoint_state_dict(self, state_dict):
        for wrapper_key in ('state_dict', 'generator', 'netG'):
            if wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
                state_dict = state_dict[wrapper_key]
                break
        model_keys = set(super().state_dict().keys())
        normalized = {}
        for key, value in state_dict.items():
            normalized_key = key
            while (normalized_key not in model_keys
                   and normalized_key.startswith(('module.', 'generator.'))):
                normalized_key = normalized_key.split('.', 1)[1]
            normalized[normalized_key] = value
        return normalized

    def _load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint = self._normalize_checkpoint_state_dict(checkpoint)
        current = super().state_dict()
        compatible = {
            key: value for key, value in checkpoint.items()
            if key in current
            and torch.is_tensor(value)
            and value.shape == current[key].shape
        }
        result = super().load_state_dict(compatible, strict=False)
        print(f'Loaded {len(compatible)}/{len(current)} generator weights')
        if result.missing_keys:
            print(f'Missing generator keys: {len(result.missing_keys)}')

    def forward(self, masked_frames, completed_flows, masks_in,
                masks_updated, num_local_frames, trajectory_result=None,
                trajectory_context_frames=None, trajectory_context_masks=None,
                trajectory_context_local_indices=None):
        local_length = int(num_local_frames)
        batch, total_frames, _, original_h, original_w = masked_frames.shape
        encoded = self.encoder(torch.cat([
            masked_frames.reshape(batch * total_frames, 3, original_h, original_w),
            masks_in.reshape(batch * total_frames, 1, original_h, original_w),
            masks_updated.reshape(batch * total_frames, 1, original_h, original_w),
        ], dim=1))
        _, channels, feature_h, feature_w = encoded.shape
        encoded = encoded.view(batch, total_frames, channels, feature_h, feature_w)
        local_feature = encoded[:, :local_length]
        reference_feature = encoded[:, local_length:]

        flow_forward = F.interpolate(
            completed_flows[0].reshape(-1, 2, original_h, original_w),
            scale_factor=0.25,
            mode='bilinear',
            align_corners=False,
        ).view(batch, local_length - 1, 2, feature_h, feature_w) / 4.0
        flow_backward = F.interpolate(
            completed_flows[1].reshape(-1, 2, original_h, original_w),
            scale_factor=0.25,
            mode='bilinear',
            align_corners=False,
        ).view(batch, local_length - 1, 2, feature_h, feature_w) / 4.0

        mask_in = F.interpolate(
            masks_in.reshape(-1, 1, original_h, original_w),
            scale_factor=0.25,
            mode='nearest',
        ).view(batch, total_frames, 1, feature_h, feature_w)
        mask_updated = F.interpolate(
            masks_updated[:, :local_length].reshape(
                -1, 1, original_h, original_w),
            scale_factor=0.25,
            mode='nearest',
        ).view(batch, local_length, 1, feature_h, feature_w)
        local_mask = torch.cat([mask_in[:, :local_length], mask_updated], dim=2)
        _, _, local_feature, _ = self.feat_prop_module(
            local_feature, flow_forward, flow_backward, local_mask)

        mask_pool = self.max_pool(mask_in[:, :local_length].reshape(
            -1, 1, feature_h, feature_w))
        mask_pool = mask_pool.view(
            batch, local_length, 1, mask_pool.size(-2), mask_pool.size(-1))

        if trajectory_result is not None:
            if trajectory_result.get('format') != 'cotracker3_sparse_feature':
                raise ValueError('Unsupported trajectory format')
            if (trajectory_context_frames is None
                    or trajectory_context_masks is None
                    or trajectory_context_local_indices is None):
                raise ValueError('Trajectory context tensors are required')

            context_batch, context_length, _, context_h, context_w = (
                trajectory_context_frames.shape)
            if context_batch != batch or (context_h, context_w) != (
                    original_h, original_w):
                raise ValueError('Trajectory context shape mismatch')

            context_features = []
            for start in range(0, context_length, 10):
                end = min(context_length, start + 10)
                context_frames = trajectory_context_frames[:, start:end]
                context_masks = trajectory_context_masks[:, start:end]
                context_input = torch.cat([
                    context_frames.reshape(-1, 3, context_h, context_w),
                    context_masks.reshape(-1, 1, context_h, context_w),
                    context_masks.reshape(-1, 1, context_h, context_w),
                ], dim=1)
                context_features.append(self.encoder(context_input).view(
                    context_batch, end - start, channels, feature_h, feature_w))
            context_feature = torch.cat(context_features, dim=1)
            context_mask = F.interpolate(
                trajectory_context_masks.reshape(-1, 1, context_h, context_w),
                size=(feature_h, feature_w),
                mode='nearest',
            ).view(context_batch, context_length, 1, feature_h, feature_w)

            local_indices = trajectory_context_local_indices
            if local_indices.ndim == 1:
                local_indices = local_indices.unsqueeze(0)
            local_indices = local_indices.long().to(local_feature.device)
            if local_indices.shape != (batch, local_length):
                raise ValueError('Local context indices shape mismatch')

            scatter_index = local_indices[:, :, None, None, None].expand(
                batch, local_length, channels, feature_h, feature_w)
            context_feature = context_feature.scatter(
                1, scatter_index, local_feature)
            local_feature = self.longterm_trajectory_dcn(
                local_feat=local_feature,
                context_feat=context_feature,
                context_masks=context_mask,
                local_context_indices=local_indices,
                trajectory_result=trajectory_result,
            )
            context_feature = context_feature.scatter(
                1, scatter_index, local_feature)
            local_feature = local_feature + self.feature_trajectory_transformer.forward_sparse(
                context_feat=context_feature,
                trajectory=trajectory_result['trajectory'],
                visibility=trajectory_result['visibility'],
                confidence=trajectory_result['confidence'],
                context_masks=context_mask,
                local_context_indices=local_indices,
                original_size=trajectory_result['original_size'],
            )

        encoded = torch.cat([local_feature, reference_feature], dim=1)
        tokens = self.ss(encoded.view(-1, channels, feature_h, feature_w),
                         batch, (feature_h, feature_w))
        mask_pool = rearrange(mask_pool, 'b t c h w -> b t h w c').contiguous()
        tokens = self.transformers(
            tokens,
            (feature_h, feature_w),
            mask_pool,
            t_dilation=2,
        )
        tokens = self.sc(tokens, total_frames, (feature_h, feature_w))
        tokens = tokens.view(batch, total_frames, channels, feature_h, feature_w)
        encoded = encoded + tokens
        output = self.decoder(
            encoded[:, :local_length].reshape(-1, channels, feature_h, feature_w))
        return torch.tanh(output).view(
            batch, local_length, 3, original_h, original_w)
