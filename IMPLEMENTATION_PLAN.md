# IMPLEMENTATION_PLAN.md — CS3005 (PDC) Project #15

**Project:** Parallel Ray Tracing / 3D Rendering of Large Scenes  
**Student:** Muhammad Ebrahim Bin Moin (23K-0540)  
**Course:** CS3005: Parallel and Distributed Computing (PDC) — Fall 2026, FAST-NUCES  
**Core Requirement:** Hybrid (MPI + OpenMP) with Sequential, OpenMP, MPI, optional CUDA GPU benchmarking, and an interactive offline HTML GUI.

---

## 1. Project Summary & Target Hardware Profiles

This project implements a unified, physically-inspired 3D ray tracer across five execution models (**Sequential**, **OpenMP**, **MPI**, **Hybrid MPI+OpenMP**, and optional **CUDA**) backed by a single shared engine and controlled via both CLI and an offline web dashboard (`web/index.html` + `server.py`).

| Feature | Primary PC (Dev & Full Benchmark) | Demo Laptop (Portable Showcase) |
| :--- | :--- | :--- |
| **CPU** | AMD Ryzen 5 8400F (**6 cores / 12 threads**) | **2 cores / 4 threads** |
| **GPU / CUDA** | NVIDIA GeForce RTX 3050 (8 GB VRAM) | **None / CPU-only** |
| **OS / Shell** | Windows + WSL2 Ubuntu | Windows + WSL2 Ubuntu |
| **Built Binaries** | `seq`, `omp`, `mpi`, `hybrid`, `cuda` | `seq`, `omp`, `mpi`, `hybrid` |
| **Auto OpenMP Sweep** | `1, 2, 3, 4, 6, 8, 12` threads | `1, 2, 4` threads |
| **Auto Hybrid Physical Sweep** | `1×6`, `2×3`, `3×2`, `6×1` | `1×2`, `2×1` |
| **Auto Hybrid Logical Sweep** | `1×12`, `2×6`, `3×4`, `4×3`, `6×2`, `12×1` | `1×4`, `2×2`, `4×1` |
| **Demo Capability** | Live CPU + CUDA + 8K rendering | Live CPU (`seq`/`omp`/`mpi`/`hybrid`) + saved RTX 3050 / 8K gallery |

---

## 2. Directory Structure

```text
pdc-project/
├── AGENTS.md                  # Persistent rules, invariants & scaling policies
├── IMPLEMENTATION_PLAN.md     # This phased implementation roadmap
├── Makefile                   # Auto-detects nvcc; builds bin/raytracer_* targets
├── server.py                  # Zero-dependency Python 3 local server for HTML GUI
├── bin/                       # Compiled binaries (raytracer_seq, omp, mpi, hybrid, [cuda])
├── assets/
│   └── sample_mesh.obj        # Bundled 3D triangle mesh for mesh_showcase preset
├── scripts/
│   ├── setup_wsl.sh           # Checks/installs g++, openmpi, python3, optional CUDA
│   └── benchmark.py           # CLI batch runner for scaling & CSV/JSON export
├── src/
│   ├── common/
│   │   ├── stb_image_write.h  # Single-header PNG writer
│   │   ├── vec3.h             # 3D vectors, colors, ray helpers (RT_HD)
│   │   ├── ray.h              # Ray struct (RT_HD)
│   │   ├── aabb.h             # Axis-Aligned Bounding Box + slab test (RT_HD)
│   │   ├── rng.h              # Stateless PCG per-(seed, x, y, sample) PRNG (RT_HD)
│   │   ├── material.h         # Diffuse, Metal, Glass (Snell+Schlick), Emissive, Checker
│   │   ├── primitive.h        # Unified Sphere + Triangle (Möller–Trumbore) primitives
│   │   ├── obj_loader.h       # Minimal .obj parser with auto-centering & scaling
│   │   ├── bvh.h              # Flat array BVH builder + iterative stack[64] traversal
│   │   ├── scene.h            # 3 deterministic presets (large_world, studio_glass, mesh_showcase)
│   │   ├── renderer.h         # Shared iterative trace_pixel() returning RGB + ray count
│   │   └── cli_and_io.h       # Uniform CLI parser, PNG writer, heatmap, PSNR, JSON telemetry
│   ├── seq_main.cpp           # 1. Sequential baseline
│   ├── omp_main.cpp           # 2. OpenMP shared-memory backend
│   ├── mpi_main.cpp           # 3. MPI distributed-memory backend (static & dynamic)
│   ├── hybrid_main.cpp        # 4. Hybrid MPI + OpenMP backend (core requirement)
│   └── cuda_main.cu           # 5. Optional CUDA GPU backend
├── web/
│   └── index.html             # Self-contained offline interactive showcase dashboard
└── results/
    ├── benchmarks.json        # Saved PC & laptop benchmark runs
    ├── gallery.json           # Saved renders & heatmap metadata for portable demo
    └── renders/               # Output PNGs and tile-ownership heatmaps
```

---

## 3. Architecture & Shared Ray-Tracing Core (`src/common/`)

- **Dual CPU/GPU Compilation (`RT_HD`):** All common math, shading, and intersection headers define `RT_HD` (`__host__ __device__` when compiled with `nvcc`, empty under `g++`/`mpic++`).
- **Iterative Flat BVH (`bvh.h`):**
  - Builds a contiguous flat array of `BVHNode`s (longest-axis midpoint/median split) and reordered `Primitive`s so both CPU cache and GPU global memory access patterns stay compact.
  - Traverses using an explicit `int stack[64]` with zero recursion.
- **Unified Primitives (`primitive.h` & `obj_loader.h`):**
  - Supports both **Spheres** and **Triangles** (Möller–Trumbore ray-triangle intersection) inside the same BVH.
  - `.obj` loader normalizes mesh bounds and places it cleanly on the scene floor.
- **Deterministic Stateless RNG (`rng.h`):**
  - Hash-seeded PCG per `(seed, x, y, sample_idx)` guarantees reproducible anti-aliasing, soft shadows, and diffuse/metal/glass scatter across all 5 backends.
- **Three Built-In Scene Presets (`scene.h`):**
  1. `large_world` — Reflective checkerboard floor + hero glass/metal/diffuse spheres + configurable `N` objects (up to 10,000+) for "large scene" benchmarking.
  2. `studio_glass` — Dense reflective/refractive glass & mirror cluster creating heavy ray-depth variance between sky and center pixels (ideal for proving dynamic scheduling vs. static).
  3. `mesh_showcase` — Loads `assets/sample_mesh.obj` (or a user-supplied `--obj`) framed by area lights and reflective spheres.
- **Telemetry & Visualization (`cli_and_io.h`):**
  - Emits PNG render, optional **tile-ownership heatmap** (hue = MPI rank, luminance/shade = OpenMP thread or CUDA block, crisp tile grid overlay), and `JSON_RESULT:{...}` with granular stage timings and `MRays/sec`.

---

## 4. Phased Implementation Roadmap

### Phase 1: WSL2 Toolchain Setup & Build System
- [x] Check WSL2 Ubuntu packages (`g++`, `openmpi-bin`, `libopenmpi-dev`, `python3`, `make`, and optional `nvcc`).
- [x] Create `scripts/setup_wsl.sh` to verify/install missing CPU packages and report CUDA availability cleanly.
- [x] Create `Makefile` with automatic `nvcc` detection:
  - Always builds `bin/raytracer_seq`, `bin/raytracer_omp`, `bin/raytracer_mpi`, `bin/raytracer_hybrid` (`-O3 -std=c++17`, no `-ffast-math`, optional `NATIVE=1`).
  - Conditionally builds `bin/raytracer_cuda` when `nvcc` is found and `NO_CUDA` is not set.
- **Verification Criteria:**
  - `make` builds all available targets with zero errors/warnings.
  - `make clean && make NO_CUDA=1` builds the 4 CPU binaries cleanly without requiring `nvcc`.

### Phase 2: Shared Core Engine & Sequential Baseline (`raytracer_seq`)
- [x] Implement zero-dependency PNG/PPM + heatmap writer (`png_writer.h`, zlib + pure-C++ stored PNG fallback).
- [x] Implement `rt_common.h`, `vec3.h`, `ray.h`, `aabb.h`, `rng.h`, `material.h`, `primitive.h`, `obj_loader.h`.
- [x] Implement `bvh.h`, `scene.h`, `renderer.h`, and `cli_and_io.h`.
- [x] Bundle `assets/sample_mesh.obj` (512-triangle sculpted twisted torus mesh).
- [x] Implement `src/seq_main.cpp`.
- **Verification Criteria:**
  - Render all 3 presets (`large_world`, `studio_glass`, `mesh_showcase`) at `640×360` and `1280×720`.
  - Confirm crisp PNG renders, accurate BVH traversal, and valid `JSON_RESULT:{...}` output.

### Phase 3: OpenMP, MPI, and Hybrid (MPI + OpenMP) Backends
- [x] Implement `src/omp_main.cpp`:
  - Tile-based `#pragma omp parallel for` supporting `static`, `dynamic`, and `guided` schedules via `omp_set_schedule` + `schedule(runtime)`.
  - Records per-tile OpenMP thread ID for heatmap visualization.
- [x] Implement `src/mpi_main.cpp`:
  - Supports `static` (contiguous tile-row slices + `MPI_Gatherv`) and `dynamic` (self-participating Master–Worker tile queue with non-stalling scanline `MPI_Iprobe` drain + `np == 1` fast path).
  - Workers allocate only per-tile buffers (8K memory safety) and transmit RGB8 tiles to Rank 0.
  - Records per-tile MPI rank ID for heatmap visualization.
- [x] Implement `src/hybrid_main.cpp` (Core Course Requirement):
  - Initializes with `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)`; keeps MPI calls on main thread only.
  - MPI distributes tiles across processes; OpenMP parallelizes pixels/tiles inside each rank with configurable schedule and non-stalling strip `MPI_Iprobe` on Rank 0.
  - Records both `(rank, thread)` ownership for combined heatmaps.
- **Verification Criteria:**
  - Pixel accuracy check: **PSNR >= 45 dB** vs. `raytracer_seq` for `omp`, `mpi`, and `hybrid` on identical seed/scene/resolution. (`inf dB`, 0 differing bytes verified!)
  - Verify laptop-like configs (`2` threads, `2` ranks, `2×2` hybrid) AND Ryzen 5 8400F configs (`6`/`12` threads, `1×6`, `2×3`, `3×2`, `2×6`, `3×4`, `6×2`).
  - Confirm `mpirun --bind-to none --use-hwthread-cpus` achieves real multi-core scaling without core-pinning bottlenecks.

### Phase 4: Optional CUDA GPU Backend (`raytracer_cuda`)
- [x] Implement `src/cuda_main.cu`:
  - Uploads flat primitives, materials, and flat BVH nodes to device memory (`cudaMalloc` / `cudaMemcpy`).
  - Launches 2D kernel grid of `block × block` threads (default `16×16`) with shared-memory per-block ray reduction.
  - Measures `h2d_transfer`, `render_compute` (kernel), and `comm_or_d2h` separately via `cudaEventRecord`.
  - Generates hardware Streaming Multiprocessor (`%smid`) block ownership heatmap.
- **Verification Criteria:**
  - Run on RTX 3050 at 1080p (`66.13 ms`, `543 MRays/s`), 4K (`223.82 ms`, `653 MRays/s`), and 8K (`453.43 ms`, `642 MRays/s`).
  - Verify **PSNR >= 40 dB** vs. `raytracer_seq` (`42.00 dB`, `41.75 dB`, `59.87 dB`).
  - Confirm `make NO_CUDA=1` and CPU-only runs remain unaffected.

### Phase 5: Interactive Offline HTML GUI (`web/index.html` + `server.py`)
- [x] Implement `server.py` (Python 3 stdlib only, `localhost:8000`):
  - `GET /api/system`: Detects CPU model, physical cores, logical threads, compiled binaries, CUDA GPU availability, and generates adaptive scaling presets for the current machine.
  - `POST /api/render`: Executes selected binary (with `--bind-to none` / `--use-hwthread-cpus` for MPI/Hybrid), returns render PNG URL, heatmap PNG URL, live log, and `timings_ms`.
  - `POST /api/benchmark`: Runs multi-backend comparisons, OpenMP schedule sweeps, MPI static vs. dynamic sweeps, and adaptive Hybrid `Ranks × Threads` matrices; persists snapshots to `results/benchmarks.json`.
  - `GET /api/gallery`: Lists saved renders and snapshots from `results/gallery.json`.
  - `POST /api/compare`: Computes PSNR/MSE between any rendered output and Sequential baseline.
- [x] Implement `web/index.html` (100% offline, zero external CDNs):
  - **Tab 1 — Live Render Studio:** Scene/resolution/backend controls, auto-scaled core/rank selectors, live execution logs, interactive **Render ↔ Tile Heatmap Split Slider**, and timing breakdown cards.
  - **Tab 2 — Benchmark & Scaling Lab:** One-click Compare All, Strong Scaling, OpenMP Schedule Comparison, MPI Static vs. Dynamic, and Hybrid `Ranks × Threads` Matrix; offline SVG charts for Time (Linear/Log10), Speedup $S(p)$ & Efficiency $E(p)$, Amdahl's Law stage breakdown, CSV export, and Correctness (PSNR) table.
  - **Tab 3 — Saved 4K/8K & RTX 3050 Gallery:** Portable demo view with split-slider modal so the laptop showcases desktop RTX 3050 benchmarks and 8K renders cleanly alongside live laptop runs.
  - **Tab 4 — PDC Architecture Explainer:** Visual diagram & viva cheat-sheet of MPI tile decomposition + OpenMP persistent teams + CUDA `%smid` blocks.
- **Verification Criteria:**
  - Verified `server.py` stdlib-only compile, 0 external URLs in `web/index.html`, all API routes (`/api/system`, `/api/render`, `/api/benchmark`, `/api/gallery`, `/api/compare`), and `NO_CUDA=1` simulated laptop mode.

### Phase 6: Full Benchmarking, Portable Demo Testing & Viva Readiness
- [x] Create `scripts/benchmark.py` for automated CLI sweeps and CSV/Markdown report table generation.
- [x] Run full benchmark suite on Primary PC (Ryzen 5 8400F 6C/12T + RTX 3050 8GB) across 1080p, 4K, and 8K, saving snapshots in `results/`.
- [x] Verify portable 2C/4T laptop profile behavior (`2` threads, `2` ranks, `1×2`, `2×1`, `2×2` without CUDA) and confirm saved PC gallery/benchmarks display properly in the HTML GUI.

---

## 5. Verification Log

| Phase | Status | Notes |
| :--- | :---: | :--- |
| Phase 1 — Toolchain & Build System | Done | WSL2 Ubuntu 26.04, GCC 15.2, OpenMPI, CUDA 12.4 (`sm_86`), `NO_CUDA=1` verified |
| Phase 2 — Shared Core & Sequential | Done | Shared `RT_HD` headers, flat iterative BVH, 3 presets + `.obj`, `seq` verified 640x360 & 1280x720 |
| Phase 3 — OpenMP, MPI, Hybrid | Done | Bit-exact (`inf dB`, 0 diffs) vs `seq`; persistent-team Hybrid `3×4` matches `omp 12T` at 1080p (`2779 ms` vs `2840 ms`, `13.38 MRays/s`); `mpi 6R dyn` 5.27x at 720p |
| Phase 4 — Optional CUDA Backend | Done | RTX 3050 `sm_86`: PSNR 41.75–59.87 dB; 1080p `66.1 ms` (40.3× vs 12T OMP), 4K `223.8 ms`, 8K `453.4 ms` (`642 MRays/s`); `NO_CUDA=1` verified |
| Phase 5 — Offline HTML GUI | Done | Zero-dependency `server.py` + `web/index.html` (0 external URLs), all REST routes + PSNR + `NO_CUDA=1` laptop mode verified |
| Phase 6 — Full Benchmarks & Demo Check | Done | `scripts/benchmark.py` ran all 7 suites (`compare_all`, `strong_scaling_omp`, `omp_schedules`, `mpi_static_vs_dynamic`, `hybrid_matrix`, `resolution_scaling` 1080p/4K/8K, `cyberpunk_showcase`): Hybrid `3×4` 9.00× (`14.22 MRays/s`), CUDA `278.78×` (`456.3 MRays/s`, 75.47 dB) on 720p `large_world`, CUDA 8K `510.3 ms` (`588.5 MRays/s`), Cyberpunk City 720p CUDA `187.11×` (`62.32 dB`) & 4K 24 SPP `8.46 s`; 6 showcase renders + heatmaps pinned in `results/gallery.json`; `NO_CUDA=1` build, 2C/4T exact PSNR (`2T/4T/2R/4R/1×2/2×1/2×2`), and `server.py --no-cuda` verified |
