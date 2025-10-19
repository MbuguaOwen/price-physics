from __future__ import annotations


import sys

try:
    from tqdm.auto import tqdm  # rich bar in terminals, simple in logs/CI
except Exception:  # fail-safe no-op
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable if iterable is not None else _Dummy()
    class _Dummy:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, *a, **k): pass
        def set_postfix_str(self, *a, **k): pass
        def close(self): pass

def pbar(total:int|None=None, desc:str="", position:int=0, leave:bool=True):
    """
    Smart tqdm with good defaults for Windows PowerShell + logs.
    - Disables fancy animations if stdout is not a TTY (keeps logs clean)
    - Adaptive columns, smooth ETA
    """
    return tqdm(
        total=total,
        desc=desc,
        unit="it",
        position=position,
        leave=leave,
        dynamic_ncols=True,
        smoothing=0.2,
        mininterval=0.5,
        disable=not sys.stdout.isatty()
    )

