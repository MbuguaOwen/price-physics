"""
Deterministic seeds (best-effort). Import and call set_global_determinism() early in training.
"""
import os
import random

import numpy as np


def set_global_determinism(seed: int = 1337) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)

