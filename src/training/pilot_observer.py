"""Epoch resource instrumentation and cooperative stopping for a bounded pilot."""
import hashlib
import resource
import time
from pathlib import Path


class PilotPause(SystemExit):
    """Stop this worker only; the common trainer preserves its last full epoch."""


class PilotObserver:
    def __init__(self, guard=None, provenance=None):
        self.guard = guard or (lambda: None)
        self.record = provenance or {"mode": "cpu_synthetic_test"}

    def provenance(self):
        code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # Renewing a resource reservation does not alter model updates. Preserve
        # its audit trail while keeping instrumentation/code/capacity bound.
        identity = {key: self.record.get(key) for key in ("mode", "stage", "launcher_sha256", "memory_cap_bytes")}
        identity["instrumentation_sha256"] = code_hash
        return {**self.record, "instrumentation_sha256": code_hash, "resume_identity": identity,
                "metrics": "unchanged common trainer; per-epoch memory peaks are added"}

    def before_batch(self, epoch):
        self.guard()

    def start_epoch(self, epoch, device):
        self.guard()
        self.started = time.perf_counter()
        if str(device).startswith("cuda"):
            import torch
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def end_epoch(self, epoch, device):
        import torch
        gpu = str(device).startswith("cuda")
        if gpu:
            torch.cuda.synchronize(device)
        return {"epoch_wall_seconds": time.perf_counter() - self.started,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if gpu else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if gpu else None,
                "memory_scope": "CUDA peak resets each epoch; RSS is process lifetime high-water mark"}
