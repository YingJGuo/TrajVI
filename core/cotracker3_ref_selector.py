import os
import sys
from pathlib import Path

import torch


class CoTracker3RefSelector:
    def __init__(self, checkpoint_path, repo_path, device='cuda',
                 retrieval_window=60, query_batch_size=512, iterations=2):
        if retrieval_window <= 0 or query_batch_size <= 0 or iterations <= 0:
            raise ValueError('retrieval_window, query_batch_size and iterations must be positive')
        self.checkpoint_path = checkpoint_path
        if repo_path:
            repo = Path(repo_path).expanduser()
            if not repo.is_absolute():
                repo = Path.cwd() / repo
        else:
            repo = Path(__file__).resolve().parents[1] / 'third_party' / 'co_tracker'
        self.repo_path = repo.resolve()
        self.device = torch.device(device)
        self.retrieval_window = int(retrieval_window)
        self.query_batch_size = int(query_batch_size)
        self.iterations = int(iterations)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            repo_path = os.path.realpath(str(self.repo_path))
            expected_root = os.path.join(repo_path, 'cotracker')
            if not os.path.isdir(expected_root):
                raise FileNotFoundError(
                    f'CoTracker3 package was not found at {expected_root}')
            paths = []
            for path_entry in sys.path:
                resolved = os.path.realpath(path_entry or os.getcwd())
                if resolved != repo_path:
                    paths.append(path_entry)
            sys.path[:] = [repo_path] + paths

            loaded = sys.modules.get('cotracker')
            if loaded is not None:
                module_file = os.path.realpath(
                    getattr(loaded, '__file__', '') or '')
                if not module_file.startswith(os.path.join(repo_path, 'cotracker')):
                    for name in list(sys.modules):
                        if name == 'cotracker' or name.startswith('cotracker.'):
                            del sys.modules[name]

            from cotracker.predictor import CoTrackerPredictor

            predictor_file = os.path.realpath(
                sys.modules[CoTrackerPredictor.__module__].__file__)
            if not predictor_file.startswith(expected_root):
                raise ImportError(
                    f'CoTracker3 was imported from {predictor_file}, '
                    f'expected {expected_root}')

            self._model = CoTrackerPredictor(
                checkpoint=self.checkpoint_path,
                offline=True,
                window_len=self.retrieval_window,
                iterations=self.iterations,
            ).to(self.device).eval()
            for parameter in self._model.parameters():
                parameter.requires_grad = False
        return self._model

    @torch.no_grad()
    def _run_tracking(self, video, queries):
        if queries.ndim != 2 or queries.shape[-1] != 3:
            raise ValueError('queries must have shape [N, 3]')
        if queries.shape[0] == 0:
            raise ValueError('CoTracker3 requires at least one query')

        predictor = self.model
        outputs = []
        for start in range(0, queries.shape[0], self.query_batch_size):
            query_batch = queries[start:start + self.query_batch_size].to(self.device)
            tracks, _, visibility, confidence = predictor(
                video,
                queries=query_batch.unsqueeze(0),
                grid_size=0,
                backward_tracking=False,
                return_scores=True,
                iters=self.iterations,
            )
            outputs.append((tracks, visibility, confidence))

        return tuple(torch.cat(items, dim=2) for items in zip(*outputs))
