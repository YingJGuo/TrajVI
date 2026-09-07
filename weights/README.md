# Model Weights

Place the following files on the machine used for inference and pass their
paths to infer.py:

| Argument | Weight |
| --- | --- |
| --checkpoint | TrajVI Full V4 generator checkpoint, for example gen_015000.pth |
| --raft-checkpoint | Official RAFT Things checkpoint, raft-things.pth |
| --flow-checkpoint | Recurrent flow-completion checkpoint, recurrent_flow_completion.pth |
| --cotracker-checkpoint | CoTracker3 Offline checkpoint, scaled_offline.pth |

The CoTracker3 repository is also required and is supplied separately through
--cotracker-repo. The generator checkpoint must be a Full V4 checkpoint with
the TLP and TTR parameters included.
