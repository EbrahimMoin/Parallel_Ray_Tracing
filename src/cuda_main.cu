#include "cli_and_io.h"
#include "renderer.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "       \
                      << cudaGetErrorString(err__) << "\n";                              \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static double elapsed_ms(
    const std::chrono::steady_clock::time_point& a,
    const std::chrono::steady_clock::time_point& b
) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

__global__ void render_kernel(
    SceneView view,
    int width,
    int height,
    int spp,
    int depth,
    uint32_t seed,
    uint8_t* d_rgb,
    unsigned long long* d_total_rays,
    int* d_block_sm
) {
    __shared__ unsigned long long s_rays[1024];

    int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    int block_threads = static_cast<int>(blockDim.x * blockDim.y);

    if (d_block_sm != nullptr && tid == 0) {
        unsigned int smid = 0;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        d_block_sm[blockIdx.y * gridDim.x + blockIdx.x] = static_cast<int>(smid);
    }

    uint64_t local_rays = 0;
    if (x < width && y < height) {
        size_t off = (static_cast<size_t>(y) * width + x) * 3;
        render_pixel(view, x, y, width, height, spp, depth, seed, &d_rgb[off], local_rays);
    }

    s_rays[tid] = static_cast<unsigned long long>(local_rays);
    __syncthreads();

    // Fold down to highest power-of-2 <= block_threads (handles any --block 4..32)
    int pow2 = 1;
    while ((pow2 << 1) <= block_threads) {
        pow2 <<= 1;
    }
    if (tid + pow2 < block_threads) {
        s_rays[tid] += s_rays[tid + pow2];
    }
    __syncthreads();

    for (int stride = pow2 >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_rays[tid] += s_rays[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_total_rays, s_rays[0]);
    }
}

int main(int argc, char** argv) {
    RenderConfig cfg;
    if (!parse_cli(argc, argv, cfg)) {
        return 1;
    }

    int dev_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count <= 0) {
        std::cerr << "No CUDA-capable GPU detected (" << cudaGetErrorString(dev_err) << ").\n";
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr)); // Warm up CUDA context cleanly before timings

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int bdim = cfg.block;
    if (bdim < 4) bdim = 4;
    if (bdim > 32) bdim = 32;

    using Clock = std::chrono::steady_clock;
    TimingStats tm;
    auto t_wall_start = Clock::now();

    SceneData scene;
    auto t0 = Clock::now();
    build_scene(cfg.scene, cfg.objects, cfg.seed, cfg.width, cfg.height, cfg.obj_path, scene);
    auto t1 = Clock::now();
    tm.scene_build = elapsed_ms(t0, t1);

    build_bvh(scene.prims, scene.nodes);
    auto t2 = Clock::now();
    tm.bvh_build = elapsed_ms(t1, t2);

    size_t fb_bytes = static_cast<size_t>(cfg.width) * cfg.height * 3;
    std::vector<uint8_t> framebuffer(fb_bytes, 0);

    dim3 block(static_cast<unsigned int>(bdim), static_cast<unsigned int>(bdim));
    dim3 grid(
        static_cast<unsigned int>((cfg.width + bdim - 1) / bdim),
        static_cast<unsigned int>((cfg.height + bdim - 1) / bdim)
    );
    int num_blocks = static_cast<int>(grid.x * grid.y);
    bool want_heatmap = !cfg.heatmap.empty();

    BVHNode* d_nodes = nullptr;
    Primitive* d_prims = nullptr;
    Material* d_mats = nullptr;
    uint8_t* d_rgb = nullptr;
    unsigned long long* d_total_rays = nullptr;
    int* d_block_sm = nullptr;

    size_t nodes_bytes = scene.nodes.size() * sizeof(BVHNode);
    size_t prims_bytes = scene.prims.size() * sizeof(Primitive);
    size_t mats_bytes  = scene.mats.size() * sizeof(Material);

    CUDA_CHECK(cudaMalloc(&d_nodes, nodes_bytes > 0 ? nodes_bytes : sizeof(BVHNode)));
    CUDA_CHECK(cudaMalloc(&d_prims, prims_bytes > 0 ? prims_bytes : sizeof(Primitive)));
    CUDA_CHECK(cudaMalloc(&d_mats, mats_bytes > 0 ? mats_bytes : sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&d_rgb, fb_bytes));
    CUDA_CHECK(cudaMalloc(&d_total_rays, sizeof(unsigned long long)));
    if (want_heatmap) {
        CUDA_CHECK(cudaMalloc(&d_block_sm, static_cast<size_t>(num_blocks) * sizeof(int)));
    }

    cudaEvent_t ev_h2d_s, ev_h2d_e, ev_k_s, ev_k_e, ev_d2h_s, ev_d2h_e;
    CUDA_CHECK(cudaEventCreate(&ev_h2d_s));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_e));
    CUDA_CHECK(cudaEventCreate(&ev_k_s));
    CUDA_CHECK(cudaEventCreate(&ev_k_e));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_s));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_e));

    CUDA_CHECK(cudaEventRecord(ev_h2d_s));
    if (nodes_bytes > 0) CUDA_CHECK(cudaMemcpy(d_nodes, scene.nodes.data(), nodes_bytes, cudaMemcpyHostToDevice));
    if (prims_bytes > 0) CUDA_CHECK(cudaMemcpy(d_prims, scene.prims.data(), prims_bytes, cudaMemcpyHostToDevice));
    if (mats_bytes > 0)  CUDA_CHECK(cudaMemcpy(d_mats, scene.mats.data(), mats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_total_rays, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventRecord(ev_h2d_e));

    SceneView d_view = scene.view();
    d_view.nodes = d_nodes;
    d_view.prims = d_prims;
    d_view.mats  = d_mats;

    CUDA_CHECK(cudaEventRecord(ev_k_s));
    render_kernel<<<grid, block>>>(
        d_view,
        cfg.width,
        cfg.height,
        cfg.spp,
        cfg.depth,
        cfg.seed,
        d_rgb,
        d_total_rays,
        d_block_sm
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_k_e));

    unsigned long long total_rays_ull = 0;
    std::vector<int> block_sm(want_heatmap ? static_cast<size_t>(num_blocks) : 0, 0);

    CUDA_CHECK(cudaEventRecord(ev_d2h_s));
    CUDA_CHECK(cudaMemcpy(framebuffer.data(), d_rgb, fb_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&total_rays_ull, d_total_rays, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (want_heatmap) {
        CUDA_CHECK(cudaMemcpy(block_sm.data(), d_block_sm, static_cast<size_t>(num_blocks) * sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(ev_d2h_e));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_e));

    float h2d_ms = 0.0f, kern_ms = 0.0f, d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_s, ev_h2d_e));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_k_s, ev_k_e));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_s, ev_d2h_e));

    tm.h2d_transfer   = static_cast<double>(h2d_ms);
    tm.render_compute = static_cast<double>(kern_ms);
    tm.comm_or_d2h    = static_cast<double>(d2h_ms);
    tm.total_compute  = tm.scene_build + tm.bvh_build + tm.h2d_transfer + tm.render_compute + tm.comm_or_d2h;

    CUDA_CHECK(cudaEventDestroy(ev_h2d_s));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_e));
    CUDA_CHECK(cudaEventDestroy(ev_k_s));
    CUDA_CHECK(cudaEventDestroy(ev_k_e));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_s));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_e));
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_prims));
    CUDA_CHECK(cudaFree(d_mats));
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_total_rays));
    if (d_block_sm) CUDA_CHECK(cudaFree(d_block_sm));

    auto t_io0 = Clock::now();
    if (!cfg.output.empty()) {
        ensure_parent_dir(cfg.output);
        if (!write_image_file(cfg.output, cfg.width, cfg.height, framebuffer.data())) {
            std::cerr << "Failed to write output image: " << cfg.output << "\n";
            return 1;
        }
    }
    if (want_heatmap) {
        ensure_parent_dir(cfg.heatmap);
        std::vector<int> block_thread(block_sm.size(), 0);
        int sm_count = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 20;
        write_heatmap_png(cfg.heatmap, cfg.width, cfg.height, bdim, block_sm, block_thread, sm_count, 1);
    }
    auto t_io1 = Clock::now();
    tm.image_io = elapsed_ms(t_io0, t_io1);
    tm.total_wall = elapsed_ms(t_wall_start, t_io1);

    RenderConfig out_cfg = cfg;
    out_cfg.block = bdim;
    emit_json_telemetry(
        "cuda",
        prop.name,
        out_cfg,
        1,
        1,
        static_cast<int>(scene.prims.size()),
        scene.triangle_count,
        static_cast<uint64_t>(total_rays_ull),
        tm
    );
    return 0;
}
