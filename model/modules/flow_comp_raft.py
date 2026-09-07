import argparse

import torch
import torch.nn as nn

from RAFT import RAFT


def initialize_RAFT(model_path='weights/raft-things.pth', device='cuda'):
    args = argparse.Namespace(
        raft_model=model_path,
        small=False,
        mixed_precision=False,
    )
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model, map_location='cpu'))
    model = model.module.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


class RAFT_bi(nn.Module):
    def __init__(self, model_path='weights/raft-things.pth', device='cuda'):
        super().__init__()
        self.fix_raft = initialize_RAFT(model_path, device=device)
        self.eval()

    def forward(self, frames, iters=20):
        batch, frame_count, channels, height, width = frames.shape
        with torch.no_grad():
            first = frames[:, :-1].reshape(-1, channels, height, width)
            second = frames[:, 1:].reshape(-1, channels, height, width)
            _, flow_forward = self.fix_raft(
                first, second, iters=iters, test_mode=True)
            _, flow_backward = self.fix_raft(
                second, first, iters=iters, test_mode=True)
        return (
            flow_forward.view(batch, frame_count - 1, 2, height, width),
            flow_backward.view(batch, frame_count - 1, 2, height, width),
        )
