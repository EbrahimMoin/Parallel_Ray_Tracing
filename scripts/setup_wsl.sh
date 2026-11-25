#!/usr/bin/env bash
set -euo pipefail

MODE="install"
for arg in "$@"; do
  case "$arg" in
    --check)
      MODE="check"
      ;;
    --cpu-only)
      MODE="cpu-only"
      ;;
    -h|--help)
      echo "Usage: bash scripts/setup_wsl.sh [--check | --cpu-only]"
      echo "  --check     Read-only inspection of CPU, GPU, tools, and dynamic scaling sweeps"
      echo "  --cpu-only  Install CPU toolchain only (g++, make, OpenMPI, python3), skip CUDA"
      echo "  (default)   Install CPU toolchain + nvidia-cuda-toolkit if NVIDIA GPU is detected"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo " PDC Project #15 — WSL2 Environment & Toolchain Check"
echo " Project Root: $PROJECT_ROOT"
echo "============================================================"

DISTRO="$(grep '^PRETTY_NAME=' /etc/os-release 2>/dev/null | cut -d= -f2- | tr -d '"' || echo 'Unknown Linux')"
CPU_MODEL="$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' || echo 'Unknown CPU')"
LOGICAL_THREADS="$(nproc 2>/dev/null || echo 1)"
CORES_PER_SOCKET="$(lscpu 2>/dev/null | awk -F: '/Core\(s\) per socket/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' || true)"
SOCKETS="$(lscpu 2>/dev/null | awk -F: '/Socket\(s\)/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' || echo 1)"

if [[ -n "${CORES_PER_SOCKET:-}" && "${CORES_PER_SOCKET}" =~ ^[0-9]+$ && "${SOCKETS}" =~ ^[0-9]+$ ]]; then
  PHYSICAL_CORES=$((CORES_PER_SOCKET * SOCKETS))
else
  PHYSICAL_CORES="$LOGICAL_THREADS"
fi

echo "OS / Distro     : $DISTRO"
echo "CPU Model       : $CPU_MODEL"
echo "Physical Cores  : $PHYSICAL_CORES"
echo "Logical Threads : $LOGICAL_THREADS"

GPU_INFO=""
CUDA_CAP=""
GPU_PRESENT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
  if [[ -n "$GPU_INFO" ]]; then
    GPU_PRESENT=1
    CUDA_CAP="$(echo "$GPU_INFO" | awk -F',' '{gsub(/[ \t\r]/, "", $3); print $3}')"
  fi
fi

if [[ "$GPU_PRESENT" -eq 1 ]]; then
  echo "NVIDIA GPU      : $GPU_INFO (sm_${CUDA_CAP//./})"
else
  echo "NVIDIA GPU      : Not detected (CPU-only mode)"
fi

echo ""
echo "--- Dynamic Scaling Presets (Auto-Computed for This Host) ---"
if command -v python3 >/dev/null 2>&1; then
  python3 - "$PHYSICAL_CORES" "$LOGICAL_THREADS" <<'PYEOF'
import sys

phys = max(1, int(sys.argv[1]))
logi = max(phys, int(sys.argv[2]))

# OpenMP strong-scaling milestones up to physical_cores, plus logical_threads
candidates = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64]
omp_sweep = sorted({c for c in candidates if c <= logi} | {phys, logi})
# Ensure divisors of phys and logi are included cleanly
for n in (phys, logi):
    for d in range(1, n + 1):
        if n % d == 0 and (d <= phys or d == logi):
            if d not in omp_sweep:
                omp_sweep.append(d)
omp_sweep = sorted(set(omp_sweep))

def factor_pairs(n):
    return [f"{r}x{n // r}" for r in range(1, n + 1) if n % r == 0]

print(f"OpenMP Sweep            : {omp_sweep}")
print(f"Hybrid Physical ({phys:2d})    : {', '.join(factor_pairs(phys))}")
if logi != phys:
    print(f"Hybrid Logical  ({logi:2d})    : {', '.join(factor_pairs(logi))}")
PYEOF
else
  echo "python3 missing — skipping dynamic sweep preview"
fi

echo ""
echo "--- Toolchain Status ---"
MISSING_CPU_PKGS=()

check_cmd() {
  local label="$1"
  local bin="$2"
  local pkg="$3"
  local path=""
  if path="$(command -v "$bin" 2>/dev/null)"; then
    printf "  %-10s : OK (%s)\n" "$label" "$path"
    return 0
  else
    printf "  %-10s : MISSING (package: %s)\n" "$label" "$pkg"
    if [[ -n "$pkg" ]]; then
      MISSING_CPU_PKGS+=("$pkg")
    fi
    return 1
  fi
}

CPU_OK=1
check_cmd "g++" "g++" "build-essential" || CPU_OK=0
check_cmd "make" "make" "build-essential" || CPU_OK=0
check_cmd "mpic++" "mpic++" "libopenmpi-dev" || CPU_OK=0
check_cmd "mpirun" "mpirun" "openmpi-bin" || CPU_OK=0
check_cmd "python3" "python3" "python3" || CPU_OK=0

NVCC_PATH=""
if NVCC_PATH="$(command -v nvcc 2>/dev/null)"; then
  printf "  %-10s : OK (%s)\n" "nvcc" "$NVCC_PATH"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC_PATH="/usr/local/cuda/bin/nvcc"
  printf "  %-10s : OK (%s)\n" "nvcc" "$NVCC_PATH"
else
  if [[ "$GPU_PRESENT" -eq 1 ]]; then
    printf "  %-10s : MISSING (optional GPU package: nvidia-cuda-toolkit)\n" "nvcc"
  else
    printf "  %-10s : NOT PRESENT (skipped — CPU-only machine)\n" "nvcc"
  fi
fi

# Deduplicate missing CPU packages
UNIQUE_CPU_PKGS=()
if [[ ${#MISSING_CPU_PKGS[@]} -gt 0 ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] && UNIQUE_CPU_PKGS+=("$line")
  done < <(printf '%s\n' "${MISSING_CPU_PKGS[@]}" | awk '!seen[$0]++')
fi

NEED_CUDA_INSTALL=0
if [[ "$GPU_PRESENT" -eq 1 && -z "$NVCC_PATH" && "$MODE" != "cpu-only" ]]; then
  NEED_CUDA_INSTALL=1
fi

echo ""
if [[ "$MODE" == "check" ]]; then
  if [[ "$CPU_OK" -eq 1 && "$NEED_CUDA_INSTALL" -eq 0 ]]; then
    echo "Status: All required tools for this host are installed."
    exit 0
  fi

  echo "Action needed — run inside WSL2 Ubuntu:"
  if [[ "$GPU_PRESENT" -eq 1 ]]; then
    echo "  cd \"$PROJECT_ROOT\" && bash scripts/setup_wsl.sh"
    echo "(Or for CPU-only without CUDA: bash scripts/setup_wsl.sh --cpu-only)"
  else
    echo "  cd \"$PROJECT_ROOT\" && bash scripts/setup_wsl.sh --cpu-only"
  fi

  if [[ ${#UNIQUE_CPU_PKGS[@]} -gt 0 ]]; then
    echo ""
    echo "Packages that will be installed:"
    echo "  CPU : ${UNIQUE_CPU_PKGS[*]}"
  fi
  if [[ "$NEED_CUDA_INSTALL" -eq 1 ]]; then
    echo "  CUDA: nvidia-cuda-toolkit (--no-install-recommends)"
  fi

  if [[ "$CPU_OK" -eq 0 ]]; then
    exit 1
  fi
  exit 0
fi

# Install mode
if [[ ${#UNIQUE_CPU_PKGS[@]} -eq 0 && "$NEED_CUDA_INSTALL" -eq 0 ]]; then
  echo "Everything required is already installed. Nothing to do."
  exit 0
fi

echo "Updating apt package index..."
sudo apt-get update

if [[ ${#UNIQUE_CPU_PKGS[@]} -gt 0 ]]; then
  echo "Installing CPU packages: ${UNIQUE_CPU_PKGS[*]}"
  sudo apt-get install -y "${UNIQUE_CPU_PKGS[@]}"
fi

if [[ "$NEED_CUDA_INSTALL" -eq 1 ]]; then
  echo "Installing nvidia-cuda-toolkit (--no-install-recommends to preserve WSL2 host driver)..."
  sudo apt-get install -y --no-install-recommends nvidia-cuda-toolkit
fi

echo ""
echo "Installation complete! Re-running check:"
exec bash "$0" --check
