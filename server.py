#!/usr/bin/env python3
"""
Zero-dependency local web server & benchmark orchestrator for CS3005 PDC Project #15.
Uses Python 3 standard library only; works cleanly with or without CUDA.
"""

import argparse
import datetime
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BIN_DIR = PROJECT_ROOT / "bin"
ASSETS_DIR = PROJECT_ROOT / "assets"
WEB_DIR = PROJECT_ROOT / "web"
RESULTS_DIR = PROJECT_ROOT / "results"
RENDERS_DIR = RESULTS_DIR / "renders"
BENCHMARKS_FILE = RESULTS_DIR / "benchmarks.json"
GALLERY_FILE = RESULTS_DIR / "gallery.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

DATA_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS = {}
FORCE_NO_CUDA = os.environ.get("NO_CUDA", "0") == "1"


def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def compute_presets(phys: int, logi: int) -> dict:
    phys = max(1, phys)
    logi = max(phys, logi)

    milestones = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64]
    sweep = {m for m in milestones if m <= logi} | {phys, logi}
    for n in (phys, logi):
        for d in range(1, n + 1):
            if n % d == 0 and (d <= phys or d == logi):
                sweep.add(d)
    omp_sweep = sorted(sweep)

    def factor_pairs(n: int) -> list:
        return [[r, n // r] for r in range(1, n + 1) if n % r == 0]

    hybrid_phys = factor_pairs(phys)
    hybrid_logi = factor_pairs(logi) if logi != phys else list(hybrid_phys)

    # Pick balanced interior hybrid pair (e.g., 3x4 on 12T, 2x2 on 4T)
    interior = [p for p in hybrid_logi if p[0] > 1 and p[1] > 1]
    if interior:
        best_hybrid = max(interior, key=lambda p: (min(p[0], p[1]), -abs(p[0] - p[1]), -p[0]))
    elif len(hybrid_logi) > 1:
        best_hybrid = hybrid_logi[len(hybrid_logi) // 2]
    else:
        best_hybrid = [1, logi]

    return {
        "omp_sweep": omp_sweep,
        "mpi_sweep": omp_sweep,
        "hybrid_physical": hybrid_phys,
        "hybrid_logical": hybrid_logi,
        "recommended_hybrid": best_hybrid,
    }


def detect_system() -> dict:
    cpu_model = "Unknown CPU"
    logical_threads = os.cpu_count() or 1
    physical_cores = logical_threads

    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        cores_per_sock = None
        sockets = 1
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k == "Model name" and v:
                cpu_model = v
            elif k == "CPU(s)" and v.isdigit():
                logical_threads = int(v)
            elif k == "Core(s) per socket" and v.isdigit():
                cores_per_sock = int(v)
            elif k == "Socket(s)" and v.isdigit():
                sockets = int(v)
        if cores_per_sock:
            physical_cores = max(1, cores_per_sock * sockets)
    except Exception:
        try:
            txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"model name\s*:\s*(.+)", txt)
            if m:
                cpu_model = m.group(1).strip()
        except Exception:
            pass

    binaries = {}
    for name in ("seq", "omp", "mpi", "hybrid", "cuda"):
        p = BIN_DIR / f"raytracer_{name}"
        binaries[name] = p.is_file() and os.access(p, os.X_OK)

    gpu_info = None
    if not FORCE_NO_CUDA:
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                out = subprocess.check_output(
                    [smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                ).strip()
                if out:
                    parts = [x.strip() for x in out.splitlines()[0].split(",")]
                    gpu_info = {
                        "name": parts[0] if len(parts) > 0 else "NVIDIA GPU",
                        "memory": parts[1] if len(parts) > 1 else "",
                        "compute_cap": parts[2] if len(parts) > 2 else "",
                    }
            except Exception:
                gpu_info = None

    cuda_available = (not FORCE_NO_CUDA) and bool(binaries.get("cuda")) and (gpu_info is not None)
    if FORCE_NO_CUDA:
        binaries["cuda"] = False

    objs = []
    if ASSETS_DIR.is_dir():
        objs = sorted(f"assets/{p.name}" for p in ASSETS_DIR.glob("*.obj"))

    return {
        "cpu_model": cpu_model,
        "physical_cores": physical_cores,
        "logical_threads": logical_threads,
        "gpu": gpu_info,
        "cuda_available": cuda_available,
        "no_cuda_forced": FORCE_NO_CUDA,
        "mpirun_available": shutil.which("mpirun") is not None,
        "binaries": binaries,
        "obj_assets": objs,
        "presets": compute_presets(physical_cores, logical_threads),
        "student": "Muhammad Ebrahim Bin Moin (23K-0540)",
        "course": "CS3005 PDC — Project #15: Parallel Ray Tracing",
    }


def decode_png_rgb(path: Path) -> tuple:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    pos = 8
    w = h = 0
    bit_depth = color_type = 0
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        if ctype == b"IHDR":
            w, h, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
        pos += 12 + length

    if bit_depth != 8 or color_type != 2:
        raise ValueError(f"Unsupported PNG format ({bit_depth}-bit, type {color_type})")

    raw = zlib.decompress(bytes(idat))
    stride = w * 3
    out = bytearray(h * stride)
    prev = bytearray(stride)

    def paeth(a: int, b: int, c: int) -> int:
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        return b if pb <= pc else c

    for y in range(h):
        row_start = y * (stride + 1)
        ftype = raw[row_start]
        row = bytearray(raw[row_start + 1 : row_start + 1 + stride])
        if ftype == 1:
            for i in range(3, stride):
                row[i] = (row[i] + row[i - 3]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = row[i - 3] if i >= 3 else 0
                row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = row[i - 3] if i >= 3 else 0
                b = prev[i]
                c = prev[i - 3] if i >= 3 else 0
                row[i] = (row[i] + paeth(a, b, c)) & 0xFF
        out[y * stride : (y + 1) * stride] = row
        prev = row

    return w, h, out


def compute_psnr(path_a: Path, path_b: Path) -> dict:
    w0, h0, buf_a = decode_png_rgb(path_a)
    w1, h1, buf_b = decode_png_rgb(path_b)
    if (w0, h0) != (w1, h1):
        raise ValueError(f"Dimension mismatch: {w0}x{h0} vs {w1}x{h1}")

    sq_err = 0
    max_diff = 0
    mv_a = memoryview(buf_a)
    mv_b = memoryview(buf_b)
    for i in range(len(mv_a)):
        d = abs(mv_a[i] - mv_b[i])
        if d > max_diff:
            max_diff = d
        sq_err += d * d

    mse = sq_err / len(mv_a)
    if mse == 0.0:
        return {"mse": 0.0, "psnr_db": None, "exact": True, "max_diff": 0}
    psnr = 10.0 * math.log10((255.0 * 255.0) / mse)
    return {
        "mse": round(mse, 4),
        "psnr_db": round(psnr, 2),
        "exact": False,
        "max_diff": int(max_diff),
    }


def read_json_store(path: Path, default_obj: dict) -> dict:
    with DATA_LOCK:
        if not path.is_file():
            return dict(default_obj)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default_obj)


def write_json_store(path: Path, data: dict) -> None:
    with DATA_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)


def safe_rel_result_path(rel_or_url: str) -> Path:
    clean = rel_or_url.lstrip("/")
    if clean.startswith("results/"):
        clean = clean[len("results/") :]
    target = (RESULTS_DIR / clean).resolve()
    if not target.is_relative_to(RESULTS_DIR.resolve()):
        raise ValueError("Path escapes results directory")
    return target


def sanitize_params(params: dict) -> dict:
    backend = str(params.get("backend", "hybrid")).lower()
    if backend not in ("seq", "omp", "mpi", "hybrid", "cuda"):
        backend = "hybrid"

    scene = str(params.get("scene", "large_world"))
    if scene not in (
        "large_world",
        "studio_glass",
        "mesh_showcase",
        "cyberpunk_city",
        "cyber_cathedral",
        "orbital_megastructure",
    ):
        scene = "large_world"

    def clamp_int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            val = int(params.get(key, default))
        except (TypeError, ValueError):
            val = default
        return max(lo, min(hi, val))

    obj_path = str(params.get("obj_path", "assets/sample_mesh.obj")).strip() or "assets/sample_mesh.obj"
    # Restrict obj_path inside workspace assets/ if relative
    schedule = str(params.get("schedule", "dynamic")).lower()
    if schedule not in ("static", "dynamic", "guided"):
        schedule = "dynamic"

    mpi_mode = str(params.get("mpi_mode", "dynamic")).lower()
    if mpi_mode not in ("static", "dynamic"):
        mpi_mode = "dynamic"

    return {
        "backend": backend,
        "scene": scene,
        "obj_path": obj_path,
        "width": clamp_int("width", 1280, 64, 7680),
        "height": clamp_int("height", 720, 64, 4320),
        "spp": clamp_int("spp", 4, 1, 256),
        "depth": clamp_int("depth", 6, 1, 32),
        "objects": clamp_int("objects", 500, 0, 25000),
        "seed": clamp_int("seed", 42, 0, 2**31 - 1),
        "tile": clamp_int("tile", 32, 4, 256),
        "threads": clamp_int("threads", 4, 1, 128),
        "ranks": clamp_int("ranks", 2, 1, 64),
        "schedule": schedule,
        "mpi_mode": mpi_mode,
        "block": clamp_int("block", 16, 4, 32),
        "heatmap": bool(params.get("heatmap", True)),
        "compare_seq": bool(params.get("compare_seq", False)),
    }


def build_command(cfg: dict, out_rel: str, hm_rel: str) -> tuple:
    backend = cfg["backend"]
    binary = BIN_DIR / f"raytracer_{backend}"
    if not binary.is_file():
        raise RuntimeError(f"Binary not found: {binary}. Run 'make' first.")
    if backend == "cuda" and FORCE_NO_CUDA:
        raise RuntimeError("CUDA disabled in CPU-only mode (--no-cuda / NO_CUDA=1).")

    args = [
        str(binary),
        "--scene", cfg["scene"],
        "--obj", cfg["obj_path"],
        "--width", str(cfg["width"]),
        "--height", str(cfg["height"]),
        "--spp", str(cfg["spp"]),
        "--depth", str(cfg["depth"]),
        "--objects", str(cfg["objects"]),
        "--seed", str(cfg["seed"]),
        "--tile", str(cfg["tile"]),
        "--threads", str(cfg["threads"]),
        "--schedule", cfg["schedule"],
        "--mpi-mode", cfg["mpi_mode"],
        "--block", str(cfg["block"]),
        "--output", out_rel,
    ]
    if hm_rel:
        args.extend(["--heatmap", hm_rel])

    env = os.environ.copy()
    if backend in ("omp", "hybrid"):
        env["OMP_NUM_THREADS"] = str(cfg["threads"])

    if backend in ("mpi", "hybrid"):
        mpirun = shutil.which("mpirun")
        if not mpirun:
            raise RuntimeError("mpirun not found in PATH.")
        mpi_prefix = [
            mpirun,
            "--bind-to", "none",
            "--use-hwthread-cpus",
            "-np", str(cfg["ranks"]),
        ]
        if backend == "hybrid":
            mpi_prefix.extend(["-x", f"OMP_NUM_THREADS={cfg['threads']}"])
        cmd = mpi_prefix + args
    else:
        cmd = args

    display_cmd = " ".join(
        f'"{c}"' if (" " in c) else c for c in cmd
    )
    return cmd, env, display_cmd


def run_binary(cfg: dict, out_rel: str, hm_rel: str, job_id: str = None) -> dict:
    cmd, env, display_cmd = build_command(cfg, out_rel, hm_rel)
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    if job_id:
        with JOBS_LOCK:
            if job_id in JOBS:
                if JOBS[job_id]["cancelled"]:
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    JOBS[job_id]["proc"] = proc

    stdout, stderr = proc.communicate()

    if job_id:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["proc"] = None
                if JOBS[job_id]["cancelled"]:
                    raise RuntimeError("Job cancelled by user.")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command exited with code {proc.returncode}:\n{display_cmd}\n{stderr.strip() or stdout.strip()}"
        )

    telemetry = None
    for line in stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            telemetry = json.loads(line[len("JSON_RESULT:") :])

    if telemetry is None:
        raise RuntimeError(f"Missing JSON_RESULT in output:\n{stdout}\n{stderr}")

    return {
        "telemetry": telemetry,
        "command": display_cmd,
        "output_url": f"/{out_rel}" if out_rel else "",
        "heatmap_url": f"/{hm_rel}" if hm_rel else "",
        "stderr": stderr.strip(),
    }


def append_job_log(job_id: str, msg: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job["logs"].append(msg)


def update_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job.update(fields)


def execute_render_job(job_id: str, raw_params: dict) -> dict:
    cfg = sanitize_params(raw_params)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_rel = f"results/renders/render_{cfg['backend']}_{stamp}.png"
    hm_rel = f"results/renders/heatmap_{cfg['backend']}_{stamp}.png" if cfg["heatmap"] else ""

    update_job(job_id, status="running", progress=0.1, current_step=f"Running {cfg['backend'].upper()}...")
    append_job_log(job_id, f"[{now_iso()}] Launching {cfg['backend']} ({cfg['width']}x{cfg['height']}, {cfg['spp']} spp)...")

    res = run_binary(cfg, out_rel, hm_rel, job_id=job_id)
    append_job_log(job_id, f"$ {res['command']}")
    t = res["telemetry"]["timings_ms"]
    append_job_log(
        job_id,
        f"-> Done in {t['total_compute']:.2f} ms compute ({res['telemetry']['mrays_per_sec']:.2f} MRays/s)",
    )

    psnr_info = None
    if cfg["compare_seq"]:
        if cfg["backend"] == "seq":
            psnr_info = {"mse": 0.0, "psnr_db": None, "exact": True, "max_diff": 0}
        else:
            update_job(job_id, progress=0.75, current_step="Running Sequential reference for PSNR...")
            ref_cfg = dict(cfg)
            ref_cfg["backend"] = "seq"
            ref_cfg["threads"] = 1
            ref_cfg["ranks"] = 1
            ref_rel = f"results/renders/ref_seq_{stamp}.png"
            ref_res = run_binary(ref_cfg, ref_rel, "", job_id=job_id)
            append_job_log(
                job_id,
                f"-> Sequential reference: {ref_res['telemetry']['timings_ms']['total_compute']:.2f} ms",
            )
            psnr_info = compute_psnr(PROJECT_ROOT / ref_rel, PROJECT_ROOT / out_rel)
            res["seq_baseline_ms"] = ref_res["telemetry"]["timings_ms"]["total_compute"]
            res["speedup_vs_seq"] = round(
                ref_res["telemetry"]["timings_ms"]["total_compute"] / max(0.001, t["total_compute"]), 2
            )
            psnr_txt = "inf dB (Exact)" if psnr_info["exact"] else f"{psnr_info['psnr_db']} dB"
            append_job_log(job_id, f"-> PSNR vs Sequential: {psnr_txt}, max_diff={psnr_info['max_diff']}")

    res["psnr"] = psnr_info
    res["config"] = cfg
    res["timestamp"] = now_iso()

    if raw_params.get("save_gallery"):
        pin_gallery_entry(
            {
                "title": raw_params.get("title") or f"{cfg['scene']} ({cfg['backend'].upper()} {cfg['width']}x{cfg['height']})",
                "note": raw_params.get("note", ""),
                "output_url": res["output_url"],
                "heatmap_url": res["heatmap_url"],
                "telemetry": res["telemetry"],
                "psnr": psnr_info,
            }
        )

    update_job(job_id, status="done", progress=1.0, current_step="Completed", result=res)
    return res


def build_benchmark_cases(suite: str, base_cfg: dict, sys_info: dict) -> list:
    phys = sys_info["physical_cores"]
    logi = sys_info["logical_threads"]
    presets = sys_info["presets"]
    rec_r, rec_t = presets["recommended_hybrid"]

    cases = []
    if suite == "compare_all":
        cases.append({"label": "Sequential (1×1)", "backend": "seq", "ranks": 1, "threads": 1, "total_workers": 1})
        cases.append({"label": f"OpenMP ({phys}T dyn)", "backend": "omp", "ranks": 1, "threads": phys, "schedule": "dynamic", "total_workers": phys})
        if logi != phys:
            cases.append({"label": f"OpenMP ({logi}T dyn)", "backend": "omp", "ranks": 1, "threads": logi, "schedule": "dynamic", "total_workers": logi})
        if phys >= 2:
            cases.append({"label": f"MPI ({phys}R dyn)", "backend": "mpi", "ranks": phys, "threads": 1, "mpi_mode": "dynamic", "total_workers": phys})
        else:
            cases.append({"label": "MPI (2R dyn)", "backend": "mpi", "ranks": 2, "threads": 1, "mpi_mode": "dynamic", "total_workers": 2})
        if phys >= 4:
            pr = 2
            pt = phys // 2
            cases.append({"label": f"Hybrid ({pr}×{pt} dyn)", "backend": "hybrid", "ranks": pr, "threads": pt, "mpi_mode": "dynamic", "schedule": "dynamic", "total_workers": pr * pt})
        cases.append({"label": f"Hybrid ({rec_r}×{rec_t} dyn)", "backend": "hybrid", "ranks": rec_r, "threads": rec_t, "mpi_mode": "dynamic", "schedule": "dynamic", "total_workers": rec_r * rec_t})
        if sys_info["cuda_available"]:
            cases.append({"label": "CUDA RTX GPU (16×16)", "backend": "cuda", "ranks": 1, "threads": 1, "block": 16, "total_workers": 1})

    elif suite == "strong_scaling_omp":
        cases.append({"label": "Sequential (1T)", "backend": "seq", "ranks": 1, "threads": 1, "total_workers": 1})
        for t in presets["omp_sweep"]:
            if t == 1:
                cases.append({"label": "OpenMP (1T)", "backend": "omp", "ranks": 1, "threads": 1, "schedule": "dynamic", "total_workers": 1})
            else:
                cases.append({"label": f"OpenMP ({t}T)", "backend": "omp", "ranks": 1, "threads": t, "schedule": "dynamic", "total_workers": t})

    elif suite == "omp_schedules":
        cases.append({"label": "Sequential (1T)", "backend": "seq", "ranks": 1, "threads": 1, "total_workers": 1})
        thread_counts = [phys] if logi == phys else [phys, logi]
        for t in thread_counts:
            for sched in ("static", "guided", "dynamic"):
                cases.append({"label": f"OMP {t}T {sched}", "backend": "omp", "ranks": 1, "threads": t, "schedule": sched, "total_workers": t})

    elif suite == "mpi_static_vs_dynamic":
        cases.append({"label": "Sequential (1R)", "backend": "seq", "ranks": 1, "threads": 1, "total_workers": 1})
        rank_list = [2] if phys <= 2 else [2, phys]
        if logi > phys and logi <= 12:
            rank_list.append(logi)
        for r in rank_list:
            for mode in ("static", "dynamic"):
                cases.append({"label": f"MPI {r}R {mode}", "backend": "mpi", "ranks": r, "threads": 1, "mpi_mode": mode, "total_workers": r})

    elif suite == "hybrid_matrix":
        cases.append({"label": "Sequential (1×1)", "backend": "seq", "ranks": 1, "threads": 1, "total_workers": 1})
        cases.append({"label": f"OMP ({phys}T)", "backend": "omp", "ranks": 1, "threads": phys, "schedule": "dynamic", "total_workers": phys})
        for r, t in presets["hybrid_physical"]:
            cases.append({"label": f"Hybrid {r}×{t} (phys)", "backend": "hybrid", "ranks": r, "threads": t, "mpi_mode": "dynamic", "schedule": "dynamic", "total_workers": r * t})
        if logi != phys:
            cases.append({"label": f"OMP ({logi}T)", "backend": "omp", "ranks": 1, "threads": logi, "schedule": "dynamic", "total_workers": logi})
            for r, t in presets["hybrid_logical"]:
                cases.append({"label": f"Hybrid {r}×{t} (logi)", "backend": "hybrid", "ranks": r, "threads": t, "mpi_mode": "dynamic", "schedule": "dynamic", "total_workers": r * t})
    else:
        raise ValueError(f"Unknown benchmark suite: {suite}")

    return cases


def execute_benchmark_job(job_id: str, raw_params: dict) -> dict:
    suite = str(raw_params.get("suite", "compare_all"))
    base_cfg = sanitize_params(raw_params)
    sys_info = detect_system()
    cases = build_benchmark_cases(suite, base_cfg, sys_info)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    suite_dir = RENDERS_DIR / f"bench_{suite}_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    results = []
    seq_ref_path = None
    seq_compute_ms = None

    total = len(cases)
    for idx, case in enumerate(cases):
        with JOBS_LOCK:
            if JOBS.get(job_id, {}).get("cancelled"):
                raise RuntimeError("Benchmark cancelled by user.")

        cfg = dict(base_cfg)
        cfg.update({k: v for k, v in case.items() if k in cfg})
        update_job(
            job_id,
            status="running",
            progress=idx / max(1, total),
            current_step=f"[{idx + 1}/{total}] Running {case['label']}...",
        )
        append_job_log(job_id, f"[{idx + 1}/{total}] {case['label']}...")

        out_rel = f"results/renders/bench_{suite}_{stamp}/run_{idx:02d}_{cfg['backend']}.png"
        hm_rel = f"results/renders/bench_{suite}_{stamp}/hm_{idx:02d}_{cfg['backend']}.png"
        run_res = run_binary(cfg, out_rel, hm_rel, job_id=job_id)
        tel = run_res["telemetry"]
        compute_ms = tel["timings_ms"]["total_compute"]

        if idx == 0 and cfg["backend"] == "seq":
            seq_ref_path = PROJECT_ROOT / out_rel
            seq_compute_ms = compute_ms

        psnr_info = None
        if seq_ref_path is not None:
            if cfg["backend"] == "seq":
                psnr_info = {"mse": 0.0, "psnr_db": None, "exact": True, "max_diff": 0}
            else:
                psnr_info = compute_psnr(seq_ref_path, PROJECT_ROOT / out_rel)

        speedup = round(seq_compute_ms / max(0.001, compute_ms), 2) if seq_compute_ms else 1.0
        workers = case.get("total_workers", cfg["ranks"] * cfg["threads"])
        efficiency_pct = round((speedup / max(1, workers)) * 100.0, 1) if cfg["backend"] != "cuda" else None

        entry = {
            "label": case["label"],
            "backend": cfg["backend"],
            "ranks": tel["ranks"],
            "threads": tel["threads"],
            "total_workers": workers,
            "schedule": tel["schedule"],
            "mpi_mode": tel["mpi_mode"],
            "block": tel["block"],
            "device_name": tel["device_name"],
            "total_rays": tel["total_rays"],
            "mrays_per_sec": tel["mrays_per_sec"],
            "timings_ms": tel["timings_ms"],
            "speedup": speedup,
            "efficiency_pct": efficiency_pct,
            "psnr": psnr_info,
            "output_url": run_res["output_url"],
            "heatmap_url": run_res["heatmap_url"],
            "command": run_res["command"],
        }
        results.append(entry)
        append_job_log(
            job_id,
            f"   -> compute={compute_ms:.1f} ms | render={tel['timings_ms']['render_compute']:.1f} ms | "
            f"{tel['mrays_per_sec']:.2f} MRays/s | speedup={speedup:.2f}x",
        )

    suite_record = {
        "id": f"{suite}_{stamp}",
        "suite": suite,
        "timestamp": now_iso(),
        "host_cpu": sys_info["cpu_model"],
        "physical_cores": sys_info["physical_cores"],
        "logical_threads": sys_info["logical_threads"],
        "gpu": sys_info["gpu"]["name"] if sys_info["gpu"] else "None (CPU-only)",
        "config": base_cfg,
        "results": results,
    }

    store = read_json_store(BENCHMARKS_FILE, {"suites": []})
    suites = store.get("suites", [])
    suites.insert(0, suite_record)
    store["suites"] = suites[:50]
    write_json_store(BENCHMARKS_FILE, store)

    update_job(job_id, status="done", progress=1.0, current_step="Benchmark suite complete", result=suite_record)
    return suite_record


def pin_gallery_entry(item: dict) -> dict:
    entry = {
        "id": uuid.uuid4().hex[:10],
        "timestamp": now_iso(),
        "title": str(item.get("title", "Saved Render")),
        "note": str(item.get("note", "")),
        "output_url": str(item.get("output_url", "")),
        "heatmap_url": str(item.get("heatmap_url", "")),
        "telemetry": item.get("telemetry", {}),
        "psnr": item.get("psnr"),
    }
    store = read_json_store(GALLERY_FILE, {"items": []})
    items = store.get("items", [])
    items.insert(0, entry)
    store["items"] = items[:60]
    write_json_store(GALLERY_FILE, store)
    return entry


def start_async_job(kind: str, worker_fn, payload: dict) -> tuple:
    job_id = uuid.uuid4().hex[:10]
    job = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "progress": 0.0,
        "current_step": "Starting...",
        "logs": [],
        "result": None,
        "error": None,
        "cancelled": False,
        "proc": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    def runner():
        try:
            worker_fn(job_id, payload)
        except Exception as exc:
            with JOBS_LOCK:
                j = JOBS.get(job_id)
                if j is not None:
                    j["status"] = "cancelled" if j["cancelled"] else "error"
                    j["error"] = str(exc)
                    j["logs"].append(f"ERROR: {exc}")

    if payload.get("sync"):
        runner()
        with JOBS_LOCK:
            snap = {k: v for k, v in JOBS[job_id].items() if k != "proc"}
        return job_id, snap

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return job_id, {"job_id": job_id, "status": "queued"}


class RaytracerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        # Keep console clean while still printing server startup/errors
        pass

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        if path in ("/", "/index.html"):
            html_file = WEB_DIR / "index.html"
            if not html_file.is_file():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "web/index.html missing"})
                return
            self.send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", html_file.read_bytes())
            return

        if path.startswith("/results/"):
            try:
                target = safe_rel_result_path(path)
                if not target.is_file():
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "File not found"})
                    return
                ext = target.suffix.lower()
                ctype = (
                    "image/png" if ext == ".png"
                    else "application/json; charset=utf-8" if ext == ".json"
                    else "application/octet-stream"
                )
                self.send_bytes(HTTPStatus.OK, ctype, target.read_bytes())
            except Exception as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if path == "/api/system":
            self.send_json(HTTPStatus.OK, detect_system())
            return

        if path == "/api/benchmarks":
            store = read_json_store(BENCHMARKS_FILE, {"suites": []})
            self.send_json(HTTPStatus.OK, store)
            return

        if path == "/api/gallery":
            store = read_json_store(GALLERY_FILE, {"items": []})
            self.send_json(HTTPStatus.OK, store)
            return

        if path.startswith("/api/job/"):
            job_id = path[len("/api/job/") :]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                    return
                snap = {k: v for k, v in job.items() if k != "proc"}
            self.send_json(HTTPStatus.OK, snap)
            return

        self.send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        try:
            body = self.read_json_body()
        except Exception as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON body: {exc}"})
            return

        if path == "/api/render":
            _, snap = start_async_job("render", execute_render_job, body)
            status = HTTPStatus.OK if snap.get("status") != "error" else HTTPStatus.INTERNAL_SERVER_ERROR
            self.send_json(status, snap)
            return

        if path == "/api/benchmark":
            _, snap = start_async_job("benchmark", execute_benchmark_job, body)
            status = HTTPStatus.OK if snap.get("status") != "error" else HTTPStatus.INTERNAL_SERVER_ERROR
            self.send_json(status, snap)
            return

        if path.startswith("/api/job/") and path.endswith("/cancel"):
            job_id = path[len("/api/job/") : -len("/cancel")]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                    return
                job["cancelled"] = True
                proc = job.get("proc")
                if proc is not None and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        proc.terminate()
            self.send_json(HTTPStatus.OK, {"job_id": job_id, "cancelled": True})
            return

        if path == "/api/gallery/pin":
            entry = pin_gallery_entry(body)
            self.send_json(HTTPStatus.OK, entry)
            return

        if path == "/api/compare":
            try:
                p_a = safe_rel_result_path(str(body.get("path_a", "")))
                p_b = safe_rel_result_path(str(body.get("path_b", "")))
                res = compute_psnr(p_a, p_b)
                self.send_json(HTTPStatus.OK, res)
            except Exception as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self.send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        if path.startswith("/api/benchmarks/"):
            sid = path[len("/api/benchmarks/") :]
            store = read_json_store(BENCHMARKS_FILE, {"suites": []})
            before = len(store.get("suites", []))
            store["suites"] = [s for s in store.get("suites", []) if s.get("id") != sid]
            write_json_store(BENCHMARKS_FILE, store)
            self.send_json(HTTPStatus.OK, {"deleted": before - len(store["suites"])})
            return

        if path.startswith("/api/gallery/"):
            gid = path[len("/api/gallery/") :]
            store = read_json_store(GALLERY_FILE, {"items": []})
            before = len(store.get("items", []))
            store["items"] = [g for g in store.get("items", []) if g.get("id") != gid]
            write_json_store(GALLERY_FILE, store)
            self.send_json(HTTPStatus.OK, {"deleted": before - len(store["items"])})
            return

        self.send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})


def main() -> None:
    global FORCE_NO_CUDA
    parser = argparse.ArgumentParser(description="PDC Project #15 Offline Web Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0 for WSL2 forwarding)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default 8000)")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA backend (simulate CPU-only laptop)")
    args = parser.parse_args()

    if args.no_cuda:
        FORCE_NO_CUDA = True

    sys_info = detect_system()
    server = ThreadingHTTPServer((args.host, args.port), RaytracerHandler)
    print("============================================================")
    print(" CS3005 PDC Project #15 — Offline Ray Tracing Showcase")
    print(f" URL            : http://localhost:{args.port}")
    print(f" CPU            : {sys_info['cpu_model']}")
    print(f" Cores/Threads  : {sys_info['physical_cores']}C / {sys_info['logical_threads']}T")
    if sys_info["cuda_available"]:
        print(f" CUDA GPU       : {sys_info['gpu']['name']} (Enabled)")
    else:
        print(" CUDA GPU       : Unavailable / Disabled (CPU-only mode)")
    print("============================================================", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
