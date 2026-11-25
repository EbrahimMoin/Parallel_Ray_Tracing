#!/usr/bin/env python3
"""
Automated CLI benchmarking, showcase rendering, and CSV/Markdown report generato
for CS3005 PDC Project #15 — Parallel Ray Tracing.

Zero external dependencies (Python 3 standard library only). Reuses server.py
helpers so CLI sweeps and the web UI share identical schemas and adaptive
hardware scaling rules (zero hardcoded core counts).
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import server  # noqa: E402

ALL_SUITES = [
    "compare_all",
    "strong_scaling_omp",
    "omp_schedules",
    "mpi_static_vs_dynamic",
    "hybrid_matrix",
    "resolution_scaling",
    "cyberpunk_showcase",
]

SUITE_TITLES = {
    "compare_all": "Model Comparison (Sequential vs OpenMP vs MPI vs Hybrid vs CUDA)",
    "strong_scaling_omp": "OpenMP Strong Scaling Sweep",
    "omp_schedules": "OpenMP Scheduling Policy Comparison (Load-Imbalanced Scene)",
    "mpi_static_vs_dynamic": "MPI Static vs Dynamic Tile Decomposition (Load-Imbalanced Scene)",
    "hybrid_matrix": "Hybrid MPI x OpenMP Factor Matrix (Physical & Logical Cores)",
    "resolution_scaling": "Resolution Scaling (1080p -> 4K -> 8K)",
    "cyberpunk_showcase": "Cyberpunk City Complex Scene (CPU vs GPU + 4K/8K)",
}


def build_extra_cases(suite: str, base_cfg: dict, sys_info: dict) -> list:
    phys = sys_info["physical_cores"]
    logi = sys_info["logical_threads"]
    rec_r, rec_t = sys_info["presets"]["recommended_hybrid"]
    cuda_ok = sys_info["cuda_available"]

    if suite == "resolution_scaling":
        cases = []
        res_tiers = [
            ("1080p", 1920, 1080, 8),
            ("4K", 3840, 2160, 8),
            ("8K", 7680, 4320, 4),
        ]
        for tag, w, h, spp in res_tiers:
            cases.append({
                "label": f"{tag} OMP ({logi}T dyn)",
                "scene": base_cfg["scene"],
                "width": w,
                "height": h,
                "spp": spp,
                "depth": base_cfg["depth"],
                "objects": base_cfg["objects"],
                "backend": "omp",
                "ranks": 1,
                "threads": logi,
                "schedule": "dynamic",
                "total_workers": logi,
                "res_group": tag,
            })
            cases.append({
                "label": f"{tag} Hybrid ({rec_r}x{rec_t} dyn)",
                "scene": base_cfg["scene"],
                "width": w,
                "height": h,
                "spp": spp,
                "depth": base_cfg["depth"],
                "objects": base_cfg["objects"],
                "backend": "hybrid",
                "ranks": rec_r,
                "threads": rec_t,
                "schedule": "dynamic",
                "mpi_mode": "dynamic",
                "total_workers": rec_r * rec_t,
                "res_group": tag,
            })
            if cuda_ok:
                cases.append({
                    "label": f"{tag} CUDA RTX (16x16)",
                    "scene": base_cfg["scene"],
                    "width": w,
                    "height": h,
                    "spp": spp,
                    "depth": base_cfg["depth"],
                    "objects": base_cfg["objects"],
                    "backend": "cuda",
                    "ranks": 1,
                    "threads": 1,
                    "block": 16,
                    "total_workers": 1,
                    "res_group": tag,
                })
        return cases

    if suite == "cyberpunk_showcase":
        common = {
            "scene": "cyberpunk_city",
            "width": 1280,
            "height": 720,
            "spp": 4,
            "depth": 12,
            "objects": 2500,
        }
        cases = [
            {**common, "label": "720p Sequential (1x1)", "backend": "seq", "ranks": 1, "threads": 1, "total_workers": 1},
            {**common, "label": f"720p OpenMP ({phys}T dyn)", "backend": "omp", "ranks": 1, "threads": phys, "schedule": "dynamic", "total_workers": phys},
        ]
        if logi != phys:
            cases.append({**common, "label": f"720p OpenMP ({logi}T dyn)", "backend": "omp", "ranks": 1, "threads": logi, "schedule": "dynamic", "total_workers": logi})
        mpi_r = max(2, phys)
        cases.append({**common, "label": f"720p MPI ({mpi_r}R dyn)", "backend": "mpi", "ranks": mpi_r, "threads": 1, "mpi_mode": "dynamic", "total_workers": mpi_r})
        cases.append({**common, "label": f"720p Hybrid ({rec_r}x{rec_t} dyn)", "backend": "hybrid", "ranks": rec_r, "threads": rec_t, "schedule": "dynamic", "mpi_mode": "dynamic", "total_workers": rec_r * rec_t})
        if cuda_ok:
            cases.append({**common, "label": "720p CUDA RTX (16x16)", "backend": "cuda", "ranks": 1, "threads": 1, "block": 16, "total_workers": 1})
            cases.append({
                "label": "4K CUDA RTX (16 SPP)",
                "scene": "cyberpunk_city",
                "width": 3840,
                "height": 2160,
                "spp": 16,
                "depth": 12,
                "objects": 2500,
                "backend": "cuda",
                "ranks": 1,
                "threads": 1,
                "block": 16,
                "total_workers": 1,
            })
            cases.append({
                "label": "8K CUDA RTX (4 SPP)",
                "scene": "cyberpunk_city",
                "width": 7680,
                "height": 4320,
                "spp": 4,
                "depth": 12,
                "objects": 2500,
                "backend": "cuda",
                "ranks": 1,
                "threads": 1,
                "block": 16,
                "total_workers": 1,
            })
        return cases

    return server.build_benchmark_cases(suite, base_cfg, sys_info)


def run_suite(suite: str, base_cfg: dict, imbalance_scene: str, sys_info: dict, tmp_root: Path) -> dict:
    suite_cfg = dict(base_cfg)
    if suite in ("omp_schedules", "mpi_static_vs_dynamic"):
        suite_cfg["scene"] = imbalance_scene
    elif suite == "cyberpunk_showcase":
        suite_cfg["scene"] = "cyberpunk_city"
        suite_cfg["width"] = 1280
        suite_cfg["height"] = 720
        suite_cfg["spp"] = 4
        suite_cfg["depth"] = 12
        suite_cfg["objects"] = 2500

    cases = build_extra_cases(suite, suite_cfg, sys_info)
    suite_tmp = tmp_root / suite
    suite_tmp.mkdir(parents=True, exist_ok=True)

    results = []
    seq_refs = {}
    group_baselines = {}
    total = len(cases)

    print(f"\n=== Running suite: {suite} ({total} cases) ===", flush=True)
    for idx, case in enumerate(cases):
        cfg = dict(suite_cfg)
        for k in ("scene", "width", "height", "spp", "depth", "objects", "seed", "tile",
                  "backend", "ranks", "threads", "schedule", "mpi_mode", "block"):
            if k in case:
                cfg[k] = case[k]

        out_path = suite_tmp / f"run_{idx:02d}_{cfg['backend']}.png"
        hm_path = suite_tmp / f"hm_{idx:02d}_{cfg['backend']}.png"
        out_rel = out_path.relative_to(PROJECT_ROOT).as_posix()
        hm_rel = hm_path.relative_to(PROJECT_ROOT).as_posix()

        print(f"  [{idx + 1}/{total}] {case['label']} ({cfg['scene']} {cfg['width']}x{cfg['height']} {cfg['spp']}spp)...", end=" ", flush=True)
        run_res = server.run_binary(cfg, out_rel, hm_rel)
        tel = run_res["telemetry"]
        tm = tel["timings_ms"]
        compute_ms = tm["total_compute"]

        ref_key = (cfg["scene"], cfg["width"], cfg["height"], cfg["spp"], cfg["depth"], cfg["objects"], cfg["seed"])
        if cfg["backend"] == "seq":
            seq_refs[ref_key] = (out_path, compute_ms)

        psnr_info = None
        seq_compute_ms = None
        if ref_key in seq_refs:
            seq_path, seq_compute_ms = seq_refs[ref_key]
            if cfg["backend"] == "seq":
                psnr_info = {"mse": 0.0, "psnr_db": None, "exact": True, "max_diff": 0}
            else:
                psnr_info = server.compute_psnr(seq_path, out_path)

        res_group = case.get("res_group")
        if res_group and res_group not in group_baselines:
            group_baselines[res_group] = compute_ms

        if seq_compute_ms:
            speedup = round(seq_compute_ms / max(0.001, compute_ms), 2)
            baseline_kind = "seq"
        elif res_group and res_group in group_baselines:
            speedup = round(group_baselines[res_group] / max(0.001, compute_ms), 2)
            baseline_kind = "omp_logical"
        else:
            speedup = None
            baseline_kind = "none"

        workers = case.get("total_workers", cfg["ranks"] * cfg["threads"])
        efficiency_pct = (
            round((speedup / max(1, workers)) * 100.0, 1)
            if (speedup is not None and baseline_kind == "seq" and cfg["backend"] != "cuda")
            else None
        )

        psnr_txt = "exact" if (psnr_info and psnr_info["exact"]) else (f"{psnr_info['psnr_db']}dB" if psnr_info else "n/a")
        sp_txt = f"{speedup:.2f}x" if speedup is not None else "n/a"
        print(f"{compute_ms:.1f} ms | {tel['mrays_per_sec']:.2f} MRays/s | {sp_txt} | PSNR={psnr_txt}", flush=True)

        results.append({
            "label": case["label"],
            "scene": cfg["scene"],
            "width": cfg["width"],
            "height": cfg["height"],
            "spp": cfg["spp"],
            "depth": cfg["depth"],
            "objects": cfg["objects"],
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
            "timings_ms": tm,
            "speedup": speedup,
            "speedup_baseline": baseline_kind,
            "efficiency_pct": efficiency_pct,
            "psnr": psnr_info,
            "output_url": "",
            "heatmap_url": "",
            "command": run_res["command"],
        })

    return {
        "id": f"{suite}_{server.now_iso().replace(' ', '_').replace(':', '')}",
        "suite": suite,
        "timestamp": server.now_iso(),
        "host_cpu": sys_info["cpu_model"],
        "physical_cores": sys_info["physical_cores"],
        "logical_threads": sys_info["logical_threads"],
        "gpu": sys_info["gpu"]["name"] if sys_info["gpu"] else "None (CPU-only)",
        "config": suite_cfg,
        "results": results,
    }


def render_showcase_gallery(sys_info: dict, seed: int, reset_gallery: bool) -> list:
    cuda_ok = sys_info["cuda_available"]
    phys = sys_info["physical_cores"]
    logi = sys_info["logical_threads"]
    rec_r, rec_t = sys_info["presets"]["recommended_hybrid"]

    # Ordered from lowest display priority to highest because pin_gallery_entry
    # prepends to items[0], making the last pinned item appear first on load.
    specs = [
        {
            "slug": "studio_glass_mpi_static",
            "title": f"Studio Glass — MPI Static ({phys}R)",
            "note": "Static contiguous tile-row split showing load imbalance on glass caustics",
            "cfg": {"backend": "mpi", "scene": "studio_glass", "width": 1920, "height": 1080, "spp": 8, "depth": 10, "objects": 500, "seed": seed, "tile": 32, "ranks": max(2, phys), "threads": 1, "mpi_mode": "static"},
        },
        {
            "slug": "studio_glass_hybrid_dynamic",
            "title": f"Studio Glass — Hybrid Dynamic ({rec_r}x{rec_t})",
            "note": "Persistent OpenMP team + scanline-Iprobe dynamic tile queue eliminating tail stragglers",
            "cfg": {"backend": "hybrid", "scene": "studio_glass", "width": 1920, "height": 1080, "spp": 8, "depth": 10, "objects": 500, "seed": seed, "tile": 32, "ranks": rec_r, "threads": rec_t, "schedule": "dynamic", "mpi_mode": "dynamic"},
        },
        {
            "slug": "orbital_megastructure_showcase",
            "title": "Orbital Megastructure — 4K RTX 3050" if cuda_ok else f"Orbital Megastructure — 1080p Hybrid ({rec_r}x{rec_t})",
            "note": "Ringworld torus halo & lit megacity skyline with volumetric dusk atmosphere",
            "cfg": (
                {"backend": "cuda", "scene": "orbital_megastructure", "width": 3840, "height": 2160, "spp": 16, "depth": 10, "objects": 1200, "seed": seed, "tile": 32, "block": 16}
                if cuda_ok else
                {"backend": "hybrid", "scene": "orbital_megastructure", "width": 1920, "height": 1080, "spp": 8, "depth": 10, "objects": 1200, "seed": seed, "tile": 32, "ranks": rec_r, "threads": rec_t, "schedule": "dynamic", "mpi_mode": "dynamic"}
            ),
        },
        {
            "slug": "cyber_cathedral_showcase",
            "title": "Cyber Cathedral — 4K RTX 3050" if cuda_ok else f"Cyber Cathedral — 1080p Hybrid ({rec_r}x{rec_t})",
            "note": "Neon pillars, refractive glass core & floating monoliths in mirror hall",
            "cfg": (
                {"backend": "cuda", "scene": "cyber_cathedral", "width": 3840, "height": 2160, "spp": 16, "depth": 10, "objects": 1200, "seed": seed, "tile": 32, "block": 16}
                if cuda_ok else
                {"backend": "hybrid", "scene": "cyber_cathedral", "width": 1920, "height": 1080, "spp": 8, "depth": 10, "objects": 1200, "seed": seed, "tile": 32, "ranks": rec_r, "threads": rec_t, "schedule": "dynamic", "mpi_mode": "dynamic"}
            ),
        },
        {
            "slug": "large_world_8k",
            "title": "Large World — 8K Ultra HD (7680x4320)",
            "note": "8K showcase render (33.18M pixels) demonstrating full-resolution path tracing + SM/tile heatmap",
            "cfg": (
                {"backend": "cuda", "scene": "large_world", "width": 7680, "height": 4320, "spp": 4, "depth": 8, "objects": 1000, "seed": seed, "tile": 32, "block": 16}
                if cuda_ok else
                {"backend": "omp", "scene": "large_world", "width": 7680, "height": 4320, "spp": 4, "depth": 8, "objects": 1000, "seed": seed, "tile": 32, "threads": logi, "schedule": "dynamic"}
            ),
        },
        {
            "slug": "cyberpunk_city_4k",
            "title": "Cyberpunk City — 4K RTX 3050" if cuda_ok else f"Cyberpunk City — 1080p Hybrid ({rec_r}x{rec_t})",
            "note": "Flagship sculpted widebody sports car on rain-rippled asphalt under cyber-glyph neon signs & skyscrapers",
            "cfg": (
                {"backend": "cuda", "scene": "cyberpunk_city", "width": 3840, "height": 2160, "spp": 24, "depth": 12, "objects": 2500, "seed": seed, "tile": 32, "block": 16}
                if cuda_ok else
                {"backend": "hybrid", "scene": "cyberpunk_city", "width": 1920, "height": 1080, "spp": 8, "depth": 12, "objects": 2500, "seed": seed, "tile": 32, "ranks": rec_r, "threads": rec_t, "schedule": "dynamic", "mpi_mode": "dynamic"}
            ),
        },
    ]

    if reset_gallery:
        server.write_json_store(server.GALLERY_FILE, {"items": []})

    print(f"\n=== Rendering {len(specs)} Showcase Gallery Items ===", flush=True)
    pinned = []
    for idx, spec in enumerate(specs, start=1):
        cfg = server.sanitize_params(spec["cfg"])
        out_rel = f"results/renders/showcase_{spec['slug']}.png"
        hm_rel = f"results/renders/showcase_{spec['slug']}_hm.png"
        print(f"  [{idx}/{len(specs)}] {spec['title']} ({cfg['width']}x{cfg['height']} @ {cfg['spp']} spp)...", end=" ", flush=True)
        res = server.run_binary(cfg, out_rel, hm_rel)
        tel = res["telemetry"]
        print(f"{tel['timings_ms']['total_compute']:.1f} ms ({tel['mrays_per_sec']:.2f} MRays/s)", flush=True)

        # Deduplicate by exact title before pinning
        store = server.read_json_store(server.GALLERY_FILE, {"items": []})
        store["items"] = [it for it in store.get("items", []) if it.get("title") != spec["title"]]
        server.write_json_store(server.GALLERY_FILE, store)

        entry = server.pin_gallery_entry({
            "title": spec["title"],
            "note": spec["note"],
            "output_url": res["output_url"],
            "heatmap_url": res["heatmap_url"],
            "telemetry": tel,
            "psnr": None,
        })
        pinned.append(entry)

    return pinned


def write_csv_report(path: Path, suites: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite", "timestamp", "host_cpu", "physical_cores", "logical_threads", "gpu",
        "label", "scene", "width", "height", "spp", "depth", "objects",
        "backend", "ranks", "threads", "total_workers", "schedule", "mpi_mode", "block",
        "total_rays", "mrays_per_sec",
        "scene_build_ms", "bvh_build_ms", "h2d_ms", "render_compute_ms",
        "comm_or_d2h_ms", "image_io_ms", "total_compute_ms", "total_wall_ms",
        "speedup", "speedup_baseline", "efficiency_pct", "psnr_db", "exact_match", "max_diff",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in suites:
            for r in s["results"]:
                tm = r["timings_ms"]
                psnr = r.get("psnr") or {}
                psnr_val = "inf" if psnr.get("exact") else (psnr.get("psnr_db") if psnr.get("psnr_db") is not None else "")
                writer.writerow({
                    "suite": s["suite"],
                    "timestamp": s["timestamp"],
                    "host_cpu": s["host_cpu"],
                    "physical_cores": s["physical_cores"],
                    "logical_threads": s["logical_threads"],
                    "gpu": s["gpu"],
                    "label": r["label"],
                    "scene": r["scene"],
                    "width": r["width"],
                    "height": r["height"],
                    "spp": r["spp"],
                    "depth": r["depth"],
                    "objects": r["objects"],
                    "backend": r["backend"],
                    "ranks": r["ranks"],
                    "threads": r["threads"],
                    "total_workers": r["total_workers"],
                    "schedule": r["schedule"],
                    "mpi_mode": r["mpi_mode"],
                    "block": r["block"],
                    "total_rays": r["total_rays"],
                    "mrays_per_sec": round(r["mrays_per_sec"], 3),
                    "scene_build_ms": round(tm["scene_build"], 3),
                    "bvh_build_ms": round(tm["bvh_build"], 3),
                    "h2d_ms": round(tm["h2d_transfer"], 3),
                    "render_compute_ms": round(tm["render_compute"], 3),
                    "comm_or_d2h_ms": round(tm["comm_or_d2h"], 3),
                    "image_io_ms": round(tm["image_io"], 3),
                    "total_compute_ms": round(tm["total_compute"], 3),
                    "total_wall_ms": round(tm["total_wall"], 3),
                    "speedup": r["speedup"] if r["speedup"] is not None else "",
                    "speedup_baseline": r.get("speedup_baseline", ""),
                    "efficiency_pct": r["efficiency_pct"] if r["efficiency_pct"] is not None else "",
                    "psnr_db": psnr_val,
                    "exact_match": psnr.get("exact", ""),
                    "max_diff": psnr.get("max_diff", ""),
                })


def write_markdown_report(path: Path, suites: list, sys_info: dict, showcase_items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpu_name = sys_info["gpu"]["name"] if sys_info["gpu"] else "None (CPU-only)"
    lines = [
        "# CS3005 PDC Project #15 — Benchmark & Verification Summary",
        "",
        f"- **Generated:** {server.now_iso()}",
        f"- **Student:** {sys_info['student']}",
        f"- **Host CPU:** {sys_info['cpu_model']} ({sys_info['physical_cores']} physical cores / {sys_info['logical_threads']} logical threads)",
        f"- **GPU / CUDA:** {gpu_name}",
        "",
    ]

    for s in suites:
        title = SUITE_TITLES.get(s["suite"], s["suite"])
        cfg = s["config"]
        lines.append(f"## {title}")
        lines.append("")
        if s["suite"] not in ("resolution_scaling", "cyberpunk_showcase"):
            lines.append(
                f"Scene: `{cfg['scene']}` | Resolution: `{cfg['width']}x{cfg['height']}` | "
                f"SPP: `{cfg['spp']}` | Depth: `{cfg['depth']}` | Objects: `{cfg['objects']}`"
            )
            lines.append("")
        elif s["suite"] == "resolution_scaling":
            lines.append(
                f"Scene: `{cfg['scene']}` | Objects: `{cfg['objects']}` | Depth: `{cfg['depth']}` "
                "(Speedup measured per resolution against logical-thread OpenMP)"
            )
            lines.append("")

        lines.append("| Configuration | Resolution | SPP | Total Compute (ms) | Render (ms) | Comm/D2H (ms) | Throughput (MRays/s) | Speedup | Eff (%) | PSNR vs Seq |")
        lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in s["results"]:
            tm = r["timings_ms"]
            psnr = r.get("psnr")
            if psnr is None:
                psnr_str = "—"
            elif psnr.get("exact"):
                psnr_str = "inf (exact)"
            else:
                psnr_str = f"{psnr['psnr_db']:.2f} dB (max diff {psnr['max_diff']})"
            sp_str = f"{r['speedup']:.2f}x" if r["speedup"] is not None else "—"
            eff_str = f"{r['efficiency_pct']:.1f}%" if r["efficiency_pct"] is not None else "—"
            lines.append(
                f"| {r['label']} | {r['width']}x{r['height']} | {r['spp']} | "
                f"{tm['total_compute']:.2f} | {tm['render_compute']:.2f} | {tm['comm_or_d2h']:.2f} | "
                f"{r['mrays_per_sec']:.2f} | {sp_str} | {eff_str} | {psnr_str} |"
            )
        lines.append("")

    if showcase_items:
        lines.append("## Curated Showcase Gallery Renders")
        lines.append("")
        lines.append("| Title | Scene | Resolution | SPP | Backend | Total Compute (ms) | MRays/s | Image |")
        lines.append("| :--- | :--- | :--- | ---: | :--- | ---: | ---: | :--- |")
        for item in reversed(showcase_items):
            tel = item.get("telemetry", {})
            tm = tel.get("timings_ms", {})
            lines.append(
                f"| {item['title']} | `{tel.get('scene', '')}` | {tel.get('width', '')}x{tel.get('height', '')} | "
                f"{tel.get('spp', '')} | `{tel.get('backend', '')}` | {tm.get('total_compute', 0):.2f} | "
                f"{tel.get('mrays_per_sec', 0):.2f} | `{item.get('output_url', '')}` |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated CLI benchmark & showcase runner for CS3005 PDC Project #15")
    parser.add_argument("--suites", default="all", help="Comma-separated suites or 'all'")
    parser.add_argument("--scene", default="large_world", help="Primary scene preset (default: large_world)")
    parser.add_argument("--imbalance-scene", default="studio_glass", help="Load-imbalance scene for schedule/MPI suites")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--objects", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA (simulate CPU-only laptop)")
    parser.add_argument("--showcase", action="store_true", help="Render & pin curated flagship showcase images")
    parser.add_argument("--reset-benchmarks", action="store_true", help="Clear results/benchmarks.json before writing")
    parser.add_argument("--reset-gallery", action="store_true", help="Clear results/gallery.json before pinning showcase items")
    parser.add_argument("--csv", default="results/report_benchmarks.csv", help="CSV report output path")
    parser.add_argument("--md", default="results/report_summary.md", help="Markdown report output path")
    args = parser.parse_args()

    if args.no_cuda:
        server.FORCE_NO_CUDA = True

    sys_info = server.detect_system()
    print("============================================================")
    print(" CS3005 PDC Project #15 — CLI Benchmark Suite")
    print(f" CPU   : {sys_info['cpu_model']} ({sys_info['physical_cores']}C / {sys_info['logical_threads']}T)")
    print(f" GPU   : {sys_info['gpu']['name'] if sys_info['gpu'] else 'None / Disabled'}")
    print("============================================================", flush=True)

    if args.suites.strip().lower() == "all":
        suites_to_run = list(ALL_SUITES)
    elif args.suites.strip().lower() in ("none", ""):
        suites_to_run = []
    else:
        suites_to_run = [s.strip() for s in args.suites.split(",") if s.strip()]
        unknown = [s for s in suites_to_run if s not in ALL_SUITES]
        if unknown:
            print(f"ERROR: Unknown suite(s): {', '.join(unknown)}", file=sys.stderr)
            return 2

    base_cfg = server.sanitize_params({
        "scene": args.scene,
        "width": args.width,
        "height": args.height,
        "spp": args.spp,
        "depth": args.depth,
        "objects": args.objects,
        "seed": args.seed,
        "tile": args.tile,
    })

    if args.reset_benchmarks:
        server.write_json_store(server.BENCHMARKS_FILE, {"suites": []})

    tmp_root = server.RESULTS_DIR / "tmp_bench"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    executed_suites = []
    try:
        for suite in suites_to_run:
            record = run_suite(suite, base_cfg, args.imbalance_scene, sys_info, tmp_root)
            executed_suites.append(record)
            store = server.read_json_store(server.BENCHMARKS_FILE, {"suites": []})
            entries = [x for x in store.get("suites", []) if x.get("id") != record["id"]]
            entries.insert(0, record)
            store["suites"] = entries[:50]
            server.write_json_store(server.BENCHMARKS_FILE, store)
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    showcase_items = []
    if args.showcase:
        showcase_items = render_showcase_gallery(sys_info, args.seed, args.reset_gallery)

    if executed_suites:
        csv_path = (PROJECT_ROOT / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
        md_path = (PROJECT_ROOT / args.md) if not Path(args.md).is_absolute() else Path(args.md)
        write_csv_report(csv_path, executed_suites)
        write_markdown_report(md_path, executed_suites, sys_info, showcase_items)
        print(f"\nSaved CSV report      : {csv_path}")
        print(f"Saved Markdown report : {md_path}")

    print(f"Saved benchmarks JSON : {server.BENCHMARKS_FILE}")
    if args.showcase:
        print(f"Saved gallery JSON    : {server.GALLERY_FILE} ({len(showcase_items)} showcase items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
