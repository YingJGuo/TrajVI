import torch
import torch.nn.functional as F


def flow_warp(value, flow, interpolation='bilinear',
              padding_mode='zeros', align_corners=True):
    if value.shape[-2:] != flow.shape[1:3]:
        raise ValueError(
            f'Input size {value.shape[-2:]} does not match flow size '
            f'{flow.shape[1:3]}')
    _, _, height, width = value.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=flow.device),
        torch.arange(width, device=flow.device),
        indexing='ij',
    )
    grid = torch.stack((grid_x, grid_y), dim=2).to(value)
    grid = grid.unsqueeze(0) + flow
    grid_x = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    return F.grid_sample(
        value,
        torch.stack((grid_x, grid_y), dim=3),
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
