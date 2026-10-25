"""Read-only NVML and host telemetry for the authorized shared LightSAKT pilot.

No CUDA context, process signaling, exclusive-owner inference or allocation
changes. Memory floors/pacing retain the existing shared research safeguards.
"""
import ctypes
import resource
from datetime import datetime, timezone
from pathlib import Path


GIB = 1024 ** 3
POLICY = {"allocator_cap_bytes": 4 * GIB, "admission_free_bytes": 10 * GIB,
          "runtime_free_bytes": 6 * GIB, "host_rss_cap_bytes": 4 * GIB,
          "host_headroom_bytes": 2 * GIB, "admission_utilization_max": 75,
          "allow_unsupported_mig_utilization": True, "inter_update_pause_seconds": .1,
          "max_step_seconds": 30, "wall_time_limit_seconds": 900}


def observe(uuid):
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    def check(code):
        if code:
            raise RuntimeError(f"NVML telemetry unavailable: status {code}")
    class Memory(ctypes.Structure):
        _fields_ = [(k, ctypes.c_ulonglong) for k in ("total", "free", "used")]
    class Process(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("memory", ctypes.c_ulonglong),
                    ("gpu_instance", ctypes.c_uint), ("compute_instance", ctypes.c_uint)]
    class Utilization(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
    check(lib.nvmlInit_v2())
    try:
        handle = ctypes.c_void_p()
        check(lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)))
        memory = Memory(); check(lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
        count = ctypes.c_uint()
        fn = lib.nvmlDeviceGetComputeRunningProcesses_v2
        code = fn(handle, ctypes.byref(count), None)
        if code not in (0, 7) or count.value > 4096:
            raise RuntimeError("NVML compute-process visibility unavailable")
        processes = []
        if count.value:
            values = (Process * count.value)()
            check(fn(handle, ctypes.byref(count), values))
            processes = [{"pid": p.pid, "used_gpu_memory_bytes": p.memory} for p in values[:count.value]]
        util = Utilization()
        code = lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
        if code not in (0, 3):
            check(code)
        utilization = util.gpu if code == 0 else None
    finally:
        lib.nvmlShutdown()
    host = {}
    for base, limit_name, used_name in ((Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
                                        (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")):
        if (base / limit_name).exists():
            raw_limit = (base / limit_name).read_text().strip()
            if raw_limit != "max":
                host["cgroup_headroom_bytes"] = int(raw_limit) - int((base / used_name).read_text())
                break
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(meminfo["MemAvailable"].split()[0]) * 1024
    headroom = min(host.get("cgroup_headroom_bytes", available), available)
    return {"observed_utc": datetime.now(timezone.utc).isoformat(), "gpu_uuid": uuid,
            "total_bytes": memory.total, "free_bytes": memory.free, "used_bytes": memory.used,
            "compute_processes": processes, "pid_attribution": "unresolved; shared use; no exclusive claim",
            "utilization_gpu_percent": utilization, "utilization_status": "unsupported_mig" if code == 3 else "ok",
            "host_headroom_bytes": headroom,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def assess(sample, admission=False):
    floor = POLICY["admission_free_bytes" if admission else "runtime_free_bytes"]
    if sample["free_bytes"] < floor:
        raise RuntimeError("Shared GPU free-memory floor breached")
    if sample["host_headroom_bytes"] < POLICY["host_headroom_bytes"]:
        raise RuntimeError("Host memory headroom floor breached")
    if sample["process_peak_rss_bytes"] > POLICY["host_rss_cap_bytes"]:
        raise RuntimeError("Pilot process host RSS cap exceeded")
    if sample["utilization_status"] == "unsupported_mig":
        return "Shared MIG utilization unavailable; headroom/cap/pacing fallback applied"
    utilization = sample["utilization_gpu_percent"]
    if utilization is None or not 0 <= utilization <= 100:
        raise RuntimeError("Invalid GPU utilization telemetry")
    if admission and utilization > POLICY["admission_utilization_max"]:
        raise RuntimeError("Shared GPU is busy above admission threshold")
    return "Shared resource checks passed"
