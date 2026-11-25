# AGENTS.md — Engineering Rules & Project Invariants

> **Read this file before implementing, modifying, or benchmarking any component in this repository.**

---

## 1. Project Identity & Context

- **Course:** CS3005 — Parallel and Distributed Computing (PDC), Fall 2026, FAST-NUCES
- **Topic:** Project #15 — Parallel Ray Tracing / 3D Rendering of Large Scenes
- **Student:** Muhammad Ebrahim Bin Moin (23K-0540)
- **Required Parallel Models:**
  1. `seq` — Sequential baseline (single-threaded C++)
  2. `omp` — OpenMP (shared-memory multi-threading)
  3. `mpi` — MPI (distributed-memory tile decomposition)
  4. `hybrid` — Hybrid (MPI + OpenMP — **core course requirement**)
  5. `cuda` — CUDA GPU acceleration (**optional** benchmark when NVIDIA CUDA is available)
- **Deliverable Interface:** Both uniform **CLI binaries** and an **interactive zero-dependency HTML GUI** (`web/index.html` + `server.py`) runnable directly in the browser for live demos and visual analysis.

---

## 2. Hardware Targets & Mandatory Adaptive Scaling Rules

| Profile | Role | CPU | Physical Cores | Logical Threads | GPU / CUDA | Environment |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Primary PC** | Dev & Full Benchmarking | AMD Ryzen 5 8400F | **6** | **12** | NVIDIA GeForce RTX 3050 (8 GB VRAM) | Windows + WSL2 Ubuntu |
| **Secondary Laptop** | Portable Live Demo | Laptop CPU | **2** | **4** | None (CPU-only) | Windows + WSL2 Ubuntu |

### Mandatory Rules
1. **Zero Hardcoded Core Counts:**
   - NEVER hardcode thread counts, MPI rank counts, or hybrid combinations (e.g., never assume 8 cores).
   - Always auto-detect `physical_cores`, `logical_threads`, and `cuda_available` dynamically at runtime (`lscpu` / `nproc` and `nvidia-smi` / `raytracer_cuda` binary probe).
2. **Dynamic Scaling Sweeps:**
   - **OpenMP strong-scaling presets** must adapt to the host CPU:
     - On 6C/12T PC: `[1, 2, 3, 4, 6, 8, 12]`
     - On 2C/4T Laptop: `[1, 2, 4]`
     - Generally: generate divisors/milestones up to `physical_cores` plus `logical_threads`.
   - **Hybrid (`MPI ranks × OpenMP threads`) sweeps** must dynamically compute valid factor pairs `(R, T)` where `R * T == physical_cores` and `R * T == logical_threads`:
     - On 6C/12T PC:
       - Physical (`6`): `(1×6, 2×3, 3×2, 6×1)`
       - Logical (`12`): `(1×12, 2×6, 3×4, 4×3, 6×2, 12×1)`
     - On 2C/4T Laptop:
       - Physical (`2`): `(1×2, 2×1)`
       - Logical (`4`): `(1×4, 2×2, 4×1)`
3. **Graceful No-CUDA Portability:**
   - `Makefile` and build scripts MUST always build `raytracer_seq`, `raytracer_omp`, `raytracer_mpi`, and `raytracer_hybrid` cleanly even when `nvcc`, the CUDA Toolkit, or an NVIDIA GPU is absent.
   - `raytracer_cuda` is conditionally built ONLY when `nvcc` is present (and must support `make NO_CUDA=1` for explicit CPU-only verification).
   - `server.py` and `web/index.html` MUST run cleanly without CUDA: visually mark CUDA as unavailable on the laptop while still displaying saved desktop RTX 3050 benchmark snapshots and 4K/8K renders from `results/`.

---

## 3. Critical Parallel-Computing & OpenMPI Invariants (Never Violate)

1. **Single Shared Engine (`__host__ __device__`):**
   - All 5 backends MUST share the exact same math, camera, materials, flat BVH traversal, and shading headers inside `src/common/`.
   - Annotate shared functions/methods with `RT_HD` (`__host__ __device__` under `__CUDACC__`, empty otherwise).
2. **No Recursion in Ray Tracing or BVH Traversal:**
   - Both BVH traversal (fixed-size local stack, e.g., `int stack[64]`) and the ray bounce loop MUST be iterative so CPU and CUDA execute identically with zero stack overflow.
3. **Deterministic Stateless Per-Pixel RNG:**
   - Use a fast PCG / Wang-hash stateless PRNG seeded solely from `(seed, x, y, sample_idx)`.
   - NEVER use `std::rand()` or shared global RNG state. This eliminates lock contention and guarantees matching pixel output across Sequential, OpenMP, MPI, Hybrid, and CUDA for PSNR/MSE verification.
4. **OpenMPI Slot & Thread Binding Safety:**
   - When `ranks > physical_cores` (e.g., testing 4 ranks on the 2C/4T laptop or 12 ranks on the 6C/12T PC), pass `--use-hwthread-cpus` (or `--oversubscribe` fallback) so `mpirun` never fails with slot exhaustion.
   - In **Hybrid (MPI + OpenMP)** runs, ALWAYS pass `--bind-to none` (or explicit PE mapping) to `mpirun` alongside `OMP_NUM_THREADS=<T>` so OpenMPI's default `bind-to core` does NOT trap all OpenMP threads onto a single core.
   - In `hybrid_main.cpp`, initialize MPI with `MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided)` and keep ALL MPI calls strictly on the main thread of each rank.
5. **Non-Deadlocking, High-Utilization Dynamic MPI:**
   - `np == 1` MUST take a direct single-rank fast path (no master–worker message loop).
   - In `dynamic` master–worker mode with `np >= 2`, Rank 0 MUST participate in rendering tiles between servicing worker requests via `MPI_Iprobe` (so `np = 2` on the 2-core laptop uses both cores instead of wasting 50% of the CPU on an idle dispatcher).
6. **8K Memory Safety:**
   - An 8K (`7680×4320`) RGB8 framebuffer is ~99.5 MB (`RGBA8` is ~132.7 MB).
   - Avoid allocating full 8K `float4` framebuffers on every MPI rank; workers should only allocate memory for the tile(s) they render and send RGB8 tile buffers to Rank 0.
7. **IEEE Floating-Point & Build Flag Safety:**
   - NEVER pass `-ffast-math` to `g++`, `mpic++`, or `nvcc` — BVH AABB slab intersection relies on IEEE NaN/Inf semantics and CPU/GPU numerical consistency.
8. **Windows/WSL2 Path & Line-Ending Hygiene:**
   - The workspace path contains a space (`pdc project`): always double-quote shell path variables (`"$PROJECT_ROOT"`).
   - Ensure `Makefile` and all `scripts/*.sh` files use Unix LF (`\n`) line endings so WSL2 `bash` and `make` never fail on `\r`.

---

## 4. Uniform CLI & JSON Telemetry Contract

All 5 binaries (`raytracer_seq`, `raytracer_omp`, `raytracer_mpi`, `raytracer_hybrid`, `raytracer_cuda`) MUST accept the same CLI interface (ignoring irrelevant backend-specific flags cleanly):

### Shared CLI Flags
- `--scene <large_world|studio_glass|mesh_showcase>`
- `--obj <path>` (optional `.obj` mesh path for `mesh_showcase`)
- `--width <int>` (default e.g. `1280`)
- `--height <int>` (default e.g. `720`)
- `--spp <int>` (samples per pixel)
- `--depth <int>` (max ray bounce depth)
- `--objects <int>` (object count for procedural scenes)
- `--seed <uint32>` (deterministic RNG & scene seed)
- `--tile <int>` (tile width/height in pixels, e.g. `32`)
- `--threads <int>` (OpenMP thread count for `omp` / `hybrid`)
- `--schedule <static|dynamic|guided>` (OpenMP scheduling policy)
- `--mpi-mode <static|dynamic>` (MPI tile distribution strategy)
- `--block <int>` (CUDA 2D thread block dimension, e.g. `16` for `16×16`)
- `--output <path.png>` (rendered image output path)
- `--heatmap <path.png>` (optional tile-ownership heatmap output path)
- `--stats-json <path.json>` (optional path to write JSON telemetry)

### Standard Output Telemetry Marker
Every binary MUST print a single-line parseable JSON result prefixed with `JSON_RESULT:`:

```text
JSON_RESULT:{"backend":"hybrid","device_name":"...","width":1920,"height":1080,"spp":8,"depth":8,"objects":2000,"tile":32,"ranks":2,"threads":6,"schedule":"dynamic","mpi_mode":"dynamic","block":16,"total_rays":123456789,"mrays_per_sec":45.67,"timings_ms":{"scene_build":12.3,"bvh_build":4.5,"h2d_transfer":0.0,"render_compute":450.2,"comm_or_d2h":8.1,"image_io":22.4,"total_compute":458.3,"total_wall":497.5}}
```

- `timings_ms` MUST include:
  - `scene_build`
  - `bvh_build`
  - `h2d_transfer` (0 on CPU backends)
  - `render_compute`
  - `comm_or_d2h` (MPI communication / GPU D2H copy overhead; 0 on `seq`/`omp`)
  - `image_io`
  - `total_compute` (parallelized + communication compute time excluding I/O)
  - `total_wall` (full end-to-end wall time)

---

## 5. Web GUI & Offline Demo Rules

1. **Zero External Python Dependencies:** `server.py` MUST use Python 3 standard library ONLY (`http.server`, `subprocess`, `json`, `threading`, `os`, `shutil`, `re`, `math`) — zero `pip install` requirements.
2. **100% Offline Self-Contained UI:** `web/index.html` MUST have zero external CDN links (no external JS libraries, CSS frameworks, or fonts) so campus/viva demos work completely offline.
3. **Portable Demo Continuity:** Benchmarks and rendered images saved to `results/benchmarks.json` and `results/gallery.json` on the desktop PC (including RTX 3050 CUDA and 8K renders) MUST load cleanly in the laptop GUI even when `cuda_available` is `false`.

---

## 6. Workflow Checklist Before Every Implementation Step

- [ ] Re-read `AGENTS.md` and inspect current phase status in `IMPLEMENTATION_PLAN.md`.
- [ ] Verify no hardcoded core counts, rank counts, or CUDA-only assumptions were introduced.
- [ ] Verify shared engine headers compile cleanly with both `g++` / `mpic++` and (when available) `nvcc`.
- [ ] After building any backend, verify pixel correctness against `raytracer_seq` (target **PSNR >= 45 dB** on CPU backends, **>= 40 dB** on CUDA) using identical `--seed`, `--width`, `--height`, `--spp`, and `--scene`.
- [ ] Test both low-core laptop-profile configs (`2` threads / `2` ranks / `2×2` hybrid) and full 6C/12T Ryzen 5 8400F configs (`6`/`12` threads, `2×3`, `3×2`, `2×6`, `3×4`, `6×2`).
