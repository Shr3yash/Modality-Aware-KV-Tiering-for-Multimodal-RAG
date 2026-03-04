from __future__ import annotations

import contextlib
import time
from typing import Iterator, Optional

import torch


@contextlib.contextmanager
def cuda_timing(enable: bool = True) -> Iterator[float]:
    """Context manager that yields elapsed time in milliseconds using CUDA events."""
    if not enable or not torch.cuda.is_available():
        start = time.perf_counter()
        yield 0.0
        end = time.perf_counter()
        _ = (end - start) * 1000.0
        return

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    yield 0.0
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    _ = elapsed_ms


def memory_snapshot(device: Optional[torch.device] = None) -> dict[str, float]:
    """Return a snapshot of CUDA memory usage in megabytes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    return {"allocated_mb": float(allocated), "reserved_mb": float(reserved)}

