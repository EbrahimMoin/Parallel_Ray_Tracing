# CS3005 PDC Project #15 — Benchmark & Verification Summary

- **Generated:** 2026-11-25 17:50:24
- **Student:** Muhammad Ebrahim Bin Moin (23K-0540)
- **Host CPU:** AMD Ryzen 5 8400F 6-Core Processor (6 physical cores / 12 logical threads)
- **GPU / CUDA:** NVIDIA GeForce RTX 3050

## Model Comparison (Sequential vs OpenMP vs MPI vs Hybrid vs CUDA)

Scene: `large_world` | Resolution: `1280x720` | SPP: `8` | Depth: `8` | Objects: `1000`

| Configuration | Resolution | SPP | Total Compute (ms) | Render (ms) | Comm/D2H (ms) | Throughput (MRays/s) | Speedup | Eff (%) | PSNR vs Seq |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential (1×1) | 1280x720 | 8 | 10189.53 | 10188.35 | 0.00 | 1.64 | 1.00x | 100.0% | inf (exact) |
| OpenMP (6T dyn) | 1280x720 | 8 | 1952.92 | 1951.44 | 0.00 | 8.54 | 5.22x | 87.0% | inf (exact) |
| OpenMP (12T dyn) | 1280x720 | 8 | 1437.12 | 1435.60 | 0.00 | 11.61 | 7.09x | 59.1% | inf (exact) |
| MPI (6R dyn) | 1280x720 | 8 | 1977.09 | 1966.28 | 9.55 | 8.44 | 5.15x | 85.8% | inf (exact) |
| Hybrid (2×3 dyn) | 1280x720 | 8 | 1893.51 | 1886.21 | 6.71 | 8.81 | 5.38x | 89.7% | inf (exact) |
| Hybrid (3×4 dyn) | 1280x720 | 8 | 1223.57 | 1207.29 | 15.48 | 13.63 | 8.33x | 69.4% | inf (exact) |
| CUDA RTX GPU (16×16) | 1280x720 | 8 | 36.55 | 33.66 | 0.71 | 456.29 | 278.78x | — | 75.47 dB (max diff 41) |

## OpenMP Strong Scaling Sweep

Scene: `large_world` | Resolution: `1280x720` | SPP: `8` | Depth: `8` | Objects: `1000`

| Configuration | Resolution | SPP | Total Compute (ms) | Render (ms) | Comm/D2H (ms) | Throughput (MRays/s) | Speedup | Eff (%) | PSNR vs Seq |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential (1T) | 1280x720 | 8 | 10273.18 | 10272.02 | 0.00 | 1.62 | 1.00x | 100.0% | inf (exact) |
| OpenMP (1T) | 1280x720 | 8 | 10008.63 | 10007.53 | 0.00 | 1.67 | 1.03x | 103.0% | inf (exact) |
| OpenMP (2T) | 1280x720 | 8 | 5076.01 | 5074.74 | 0.00 | 3.29 | 2.02x | 101.0% | inf (exact) |
| OpenMP (3T) | 1280x720 | 8 | 3422.96 | 3421.58 | 0.00 | 4.87 | 3.00x | 100.0% | inf (exact) |
| OpenMP (4T) | 1280x720 | 8 | 2662.54 | 2658.37 | 0.00 | 6.26 | 3.86x | 96.5% | inf (exact) |
| OpenMP (6T) | 1280x720 | 8 | 1891.21 | 1886.78 | 0.00 | 8.82 | 5.43x | 90.5% | inf (exact) |
| OpenMP (8T) | 1280x720 | 8 | 1516.39 | 1511.50 | 0.00 | 11.00 | 6.77x | 84.6% | inf (exact) |
| OpenMP (12T) | 1280x720 | 8 | 1438.74 | 1433.57 | 0.00 | 11.59 | 7.14x | 59.5% | inf (exact) |

## OpenMP Scheduling Policy Comparison (Load-Imbalanced Scene)

Scene: `studio_glass` | Resolution: `1280x720` | SPP: `8` | Depth: `8` | Objects: `1000`

| Configuration | Resolution | SPP | Total Compute (ms) | Render (ms) | Comm/D2H (ms) | Throughput (MRays/s) | Speedup | Eff (%) | PSNR vs Seq |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential (1T) | 1280x720 | 8 | 27002.77 | 27001.40 | 0.00 | 1.12 | 1.00x | 100.0% | inf (exact) |
| OMP 6T static | 1280x720 | 8 | 7008.32 | 7003.81 | 0.00 | 4.30 | 3.85x | 64.2% | inf (exact) |
| OMP 6T guided | 1280x720 | 8 | 4845.05 | 4840.86 | 0.00 | 6.23 | 5.57x | 92.8% | inf (exact) |
| OMP 6T dynamic | 1280x720 | 8 | 5306.47 | 5302.12 | 0.00 | 5.68 | 5.09x | 84.8% | inf (exact) |
| OMP 12T static | 1280x720 | 8 | 4530.86 | 4529.47 | 0.00 | 6.66 | 5.96x | 49.7% | inf (exact) |
| OMP 12T guided | 1280x720 | 8 | 3296.79 | 3295.25 | 0.00 | 9.15 | 8.19x | 68.2% | inf (exact) |
| OMP 12T dynamic | 1280x720 | 8 | 3527.05 | 3525.74 | 0.00 | 8.55 | 7.66x | 63.8% | inf (exact) |

## MPI Static vs Dynamic Tile Decomposition (Load-Imbalanced Scene)

Scene: `studio_glass` | Resolution: `1280x720` | SPP: `8` | Depth: `8` | Objects: `1000`

| Configuration | Resolution | SPP | Total Compute (ms) | Render (ms) | Comm/D2H (ms) | Throughput (MRays/s) | Speedup | Eff (%) | PSNR vs Seq |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential (1R) | 1280x720 | 8 | 26702.83 | 26701.63 | 0.00 | 1.13 | 1.00x | 100.0% | inf (exact) |
| MPI 2R static | 1280x720 | 8 | 17981.53 | 17980.62 | 0.00 | 1.68 | 1.49x | 74.5% | inf (exact) |
| MPI 2R dynamic | 1280x720 | 8 | 13820.53 | 13778.70 | 40.94 | 2.18 | 1.93x | 96.5% | inf (exact) |
| MPI 6R static | 1280x720 | 8 | 8184.35 | 8183.42 | 0.00 | 3.69 | 3.26x | 54.3% | inf (exact) |
| MPI 6R dynamic | 1280x720 | 8 | 5271.76 | 5237.48 | 33.24 | 5.72 | 5.07x | 84.5% | inf (exact) |
| MPI 12R static | 1280x720 | 8 | 6311.02 | 6310.04 | 0.00 | 4.78 | 4.23x | 35.2% | inf (exact) |
| MPI 12R dynamic | 1280x720 | 8 | 3696.38 | 3661.50 | 32.51 | 8.16 | 7.22x | 60.2% | inf (exact) |

## Hybrid MPI x OpenMP Factor Matrix (Physical & Logical Cores)

Scene: `large_world` | Resolution: `1280x720` | SPP: `8` | Depth: `8` | Objects: `1000`

| Configuration | Resolution | SPP | Total Compute (ms) | Render (ms) | Comm/D2H (ms) | Throughput (MRays/s) | Speedup | Eff (%) | PSNR vs Seq |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential (1×1) | 1280x720 | 8 | 10552.36 | 10551.17 | 0.00 | 1.58 | 1.00x | 100.0% | inf (exact) |
| OMP (6T) | 1280x720 | 8 | 2007.37 | 2005.52 | 0.00 | 8.31 | 5.26x | 87.7% | inf (exact) |
| Hybrid 1×6 (phys) | 1280x720 | 8 | 1923.42 | 1922.38 | 0.00 | 8.67 | 5.49x | 91.5% | inf (exact) |
| Hybrid 2×3 (phys) | 1280x720 | 8 | 1978.83 | 1961.81 | 15.70 | 8.43 | 5.33x | 88.8% | inf (exact) |
| Hybrid 3×2 (phys) | 1280x720 | 8 | 2017.19 | 2003.35 | 12.87 | 8.27 | 5.23x | 87.2% | inf (exact) |
| Hybrid 6×1 (phys) | 1280x720 | 8 | 1999.78 | 1964.39 | 34.48 | 8.34 | 5.28x | 88.0% | inf (exact) |
| OMP (12T) | 1280x720 | 8 | 1359.26 | 1357.34 | 0.00 | 12.27 | 7.76x | 64.7% | inf (exact) |
| Hybrid 1×12 (logi) | 1280x720 | 8 | 1251.81 | 1249.77 | 0.00 | 13.32 | 8.43x | 70.2% | inf (exact) |
| Hybrid 2×6 (logi) | 1280x720 | 8 | 1230.14 | 1221.56 | 7.77 | 13.56 | 8.58x | 71.5% | inf (exact) |
| Hybrid 3×4 (logi) | 1280x720 | 8 | 1172.68 | 1153.99 | 17.66 | 14.22 | 9.00x | 75.0% | inf (exact) |
| Hybrid 4×3 (logi) | 1280x720 | 8 | 1347.63 | 1330.30 | 16.13 | 12.38 | 7.83x | 65.2% | inf (exact) |
| Hybrid 6×2 (logi) | 1280x720 | 8 | 1315.21 | 1283.54 | 19.52 | 12.68 | 8.02x | 66.8% | inf (exact) |
| Hybrid 12×1 (logi) | 1280x720 | 8 | 1290.23 | 1258.74 | 30.57 | 12.93 | 8.18x | 68.2% | inf (exact) |

## Resolution Scaling (1080p -> 4K -> 8K)

Scene: `large_world` | Objects: `1000` | Depth: `8` (Speedup measured per resolution against logical-thread OpenMP)

| Configuration | Resolution | SPP | Total Compute (ms) | Render (ms) | Comm/D2H (ms) | Throughput (MRays/s) | Speedup | Eff (%) | PSNR vs Seq |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p OMP (12T dyn) | 1920x1080 | 8 | 3327.96 | 3326.44 | 0.00 | 11.28 | 1.00x | — | — |
| 1080p Hybrid (3x4 dyn) | 1920x1080 | 8 | 3343.72 | 3321.49 | 21.04 | 11.22 | 1.00x | — | — |
| 1080p CUDA RTX (16x16) | 1920x1080 | 8 | 72.97 | 69.05 | 1.57 | 514.28 | 45.61x | — | — |
| 4K OMP (12T dyn) | 3840x2160 | 8 | 11926.71 | 11925.29 | 0.00 | 12.59 | 1.00x | — | — |
| 4K Hybrid (3x4 dyn) | 3840x2160 | 8 | 10958.78 | 10868.30 | 86.50 | 13.70 | 1.09x | — | — |
| 4K CUDA RTX (16x16) | 3840x2160 | 8 | 258.79 | 252.75 | 4.05 | 580.16 | 46.09x | — | — |
| 8K OMP (12T dyn) | 7680x4320 | 4 | 21098.30 | 21096.58 | 0.00 | 14.24 | 1.00x | — | — |
| 8K Hybrid (3x4 dyn) | 7680x4320 | 4 | 21142.70 | 20958.60 | 183.33 | 14.21 | 1.00x | — | — |
| 8K CUDA RTX (16x16) | 7680x4320 | 4 | 510.33 | 494.87 | 13.45 | 588.54 | 41.34x | — | — |

## Cyberpunk City Complex Scene (CPU vs GPU + 4K/8K)

| Configuration | Resolution | SPP | Total Compute (ms) | Render (ms) | Comm/D2H (ms) | Throughput (MRays/s) | Speedup | Eff (%) | PSNR vs Seq |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p Sequential (1x1) | 1280x720 | 4 | 43300.64 | 43279.51 | 0.00 | 0.28 | 1.00x | 100.0% | inf (exact) |
| 720p OpenMP (6T dyn) | 1280x720 | 4 | 7882.83 | 7859.11 | 0.00 | 1.52 | 5.49x | 91.5% | inf (exact) |
| 720p OpenMP (12T dyn) | 1280x720 | 4 | 5469.66 | 5448.37 | 0.00 | 2.19 | 7.92x | 66.0% | inf (exact) |
| 720p MPI (6R dyn) | 1280x720 | 4 | 8293.98 | 8225.43 | 46.94 | 1.45 | 5.22x | 87.0% | inf (exact) |
| 720p Hybrid (3x4 dyn) | 1280x720 | 4 | 5358.04 | 5292.50 | 43.12 | 2.24 | 8.08x | 67.3% | inf (exact) |
| 720p CUDA RTX (16x16) | 1280x720 | 4 | 231.42 | 208.44 | 0.83 | 51.85 | 187.11x | — | 62.32 dB (max diff 202) |
| 4K CUDA RTX (16 SPP) | 3840x2160 | 16 | 5735.67 | 5712.04 | 3.93 | 75.31 | — | — | — |
| 8K CUDA RTX (4 SPP) | 7680x4320 | 4 | 5905.10 | 5873.05 | 12.56 | 73.15 | — | — | — |

## Curated Showcase Gallery Renders

| Title | Scene | Resolution | SPP | Backend | Total Compute (ms) | MRays/s | Image |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | :--- |
| Cyberpunk City — 4K RTX 3050 | `cyberpunk_city` | 3840x2160 | 24 | `cuda` | 8458.58 | 76.60 | `/results/renders/showcase_cyberpunk_city_4k.png` |
| Large World — 8K Ultra HD (7680x4320) | `large_world` | 7680x4320 | 4 | `cuda` | 482.92 | 621.95 | `/results/renders/showcase_large_world_8k.png` |
| Cyber Cathedral — 4K RTX 3050 | `cyber_cathedral` | 3840x2160 | 16 | `cuda` | 1586.45 | 304.86 | `/results/renders/showcase_cyber_cathedral_showcase.png` |
| Orbital Megastructure — 4K RTX 3050 | `orbital_megastructure` | 3840x2160 | 16 | `cuda` | 330.94 | 582.90 | `/results/renders/showcase_orbital_megastructure_showcase.png` |
| Studio Glass — Hybrid Dynamic (3x4) | `studio_glass` | 1920x1080 | 8 | `hybrid` | 7362.65 | 9.86 | `/results/renders/showcase_studio_glass_hybrid_dynamic.png` |
| Studio Glass — MPI Static (6R) | `studio_glass` | 1920x1080 | 8 | `mpi` | 16558.40 | 4.38 | `/results/renders/showcase_studio_glass_mpi_static.png` |
