#include "cli_and_io.h"
#include "renderer.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <vector>

static double elapsed_ms(
    const std::chrono::steady_clock::time_point& a,
    const std::chrono::steady_clock::time_point& b
) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void configure_omp_schedule(const std::string& sched, int dyn_chunk) {
    if (sched == "static") {
        omp_set_schedule(omp_sched_static, 0);
    } else if (sched == "guided") {
        omp_set_schedule(omp_sched_guided, std::max(1, dyn_chunk / 2));
    } else {
        omp_set_schedule(omp_sched_dynamic, std::max(1, dyn_chunk));
    }
}

struct WorkBatch {
    int start_tile;
    int count;
};

struct BatchResultHeader {
    int      start_tile;
    int      count;
    uint64_t rays;
};

static void get_tile_bounds(
    int tidx, int tiles_x, int tile_size, int width, int height,
    int& x0, int& y0, int& x1, int& y1
) {
    int ty = tidx / tiles_x;
    int tx = tidx % tiles_x;
    y0 = ty * tile_size;
    y1 = rt_min(y0 + tile_size, height);
    x0 = tx * tile_size;
    x1 = rt_min(x0 + tile_size, width);
}

static void render_tile_serial_pixels(
    const SceneView& view,
    const RenderConfig& cfg,
    int x0, int y0, int x1, int y1,
    uint8_t* dst,
    int dst_stride_px,
    int dst_origin_x,
    int dst_origin_y,
    uint64_t& rays_out
) {
    uint64_t local_rays = 0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            size_t off = (static_cast<size_t>(y - dst_origin_y) * dst_stride_px + (x - dst_origin_x)) * 3;
            render_pixel(view, x, y, cfg.width, cfg.height, cfg.spp, cfg.depth, cfg.seed, &dst[off], local_rays);
        }
    }
    rays_out += local_rays;
}

static void rank_tile_rows(
    int r, int world_size, int tiles_y, int tile_size, int height,
    int& ty0, int& ty1, int& y0, int& y1
) {
    int base = tiles_y / world_size;
    int rem = tiles_y % world_size;
    ty0 = r * base + std::min(r, rem);
    ty1 = ty0 + base + (r < rem ? 1 : 0);
    y0 = std::min(ty0 * tile_size, height);
    y1 = std::min(ty1 * tile_size, height);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    RenderConfig cfg;
    if (!parse_cli(argc, argv, cfg)) {
        MPI_Finalize();
        return (rank == 0) ? 1 : 0;
    }

    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }

    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

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

    MPI_Barrier(MPI_COMM_WORLD);
    SceneView view = scene.view();

    int tiles_x = (cfg.width + cfg.tile - 1) / cfg.tile;
    int tiles_y = (cfg.height + cfg.tile - 1) / cfg.tile;
    int num_tiles = tiles_x * tiles_y;

    // Only Rank 0 allocates the full framebuffer (8K memory safety)
    std::vector<uint8_t> framebuffer;
    std::vector<int> tile_rank;
    std::vector<int> tile_thread;
    if (rank == 0) {
        framebuffer.assign(static_cast<size_t>(cfg.width) * cfg.height * 3, 0);
        tile_rank.assign(num_tiles, 0);
        tile_thread.assign(num_tiles, 0);
    }

    uint64_t total_rays = 0;
    auto t_par_start = Clock::now();

    if (world_size == 1) {
        // Direct single-rank fast path: OpenMP across tiles
        configure_omp_schedule(cfg.schedule, 1);
        #pragma omp parallel for schedule(runtime) reduction(+:total_rays)
        for (int tidx = 0; tidx < num_tiles; ++tidx) {
            tile_thread[tidx] = omp_get_thread_num();
            int x0, y0, x1, y1;
            get_tile_bounds(tidx, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
            render_tile_serial_pixels(view, cfg, x0, y0, x1, y1, framebuffer.data(), cfg.width, 0, 0, total_rays);
        }
        auto t_par_end = Clock::now();
        tm.render_compute = elapsed_ms(t_par_start, t_par_end);
        tm.comm_or_d2h = 0.0;
        tm.total_compute = tm.scene_build + tm.bvh_build + tm.render_compute;
    } else if (cfg.mpi_mode == "static") {
        // MPI contiguous tile-row slices + OpenMP across each rank's tiles
        configure_omp_schedule(cfg.schedule, 1);

        int my_ty0 = 0, my_ty1 = 0, my_y0 = 0, my_y1 = 0;
        rank_tile_rows(rank, world_size, tiles_y, cfg.tile, cfg.height, my_ty0, my_ty1, my_y0, my_y1);

        int local_rows = my_y1 - my_y0;
        int local_bytes = local_rows * cfg.width * 3;
        std::vector<uint8_t> local_buf(static_cast<size_t>(local_bytes), 0);

        int local_tile_start = my_ty0 * tiles_x;
        int local_tile_count = (my_ty1 - my_ty0) * tiles_x;
        std::vector<int> local_threads(static_cast<size_t>(local_tile_count), 0);

        uint64_t local_rays = 0;
        auto t_loc0 = Clock::now();

        #pragma omp parallel for schedule(runtime) reduction(+:local_rays)
        for (int lt = 0; lt < local_tile_count; ++lt) {
            local_threads[lt] = omp_get_thread_num();
            int tidx = local_tile_start + lt;
            int x0, y0, x1, y1;
            get_tile_bounds(tidx, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
            render_tile_serial_pixels(view, cfg, x0, y0, x1, y1, local_buf.data(), cfg.width, 0, my_y0, local_rays);
        }

        auto t_loc1 = Clock::now();
        double local_render_ms = elapsed_ms(t_loc0, t_loc1);

        std::vector<int> recvcounts, displs, t_counts, t_displs;
        if (rank == 0) {
            recvcounts.resize(world_size);
            displs.resize(world_size);
            t_counts.resize(world_size);
            t_displs.resize(world_size);
            for (int r = 0; r < world_size; ++r) {
                int r_ty0, r_ty1, r_y0, r_y1;
                rank_tile_rows(r, world_size, tiles_y, cfg.tile, cfg.height, r_ty0, r_ty1, r_y0, r_y1);
                recvcounts[r] = (r_y1 - r_y0) * cfg.width * 3;
                displs[r] = r_y0 * cfg.width * 3;
                t_counts[r] = (r_ty1 - r_ty0) * tiles_x;
                t_displs[r] = r_ty0 * tiles_x;
                for (int ty = r_ty0; ty < r_ty1; ++ty) {
                    for (int tx = 0; tx < tiles_x; ++tx) {
                        tile_rank[ty * tiles_x + tx] = r;
                    }
                }
            }
        }

        MPI_Gatherv(
            local_buf.data(), local_bytes, MPI_BYTE,
            rank == 0 ? framebuffer.data() : nullptr,
            rank == 0 ? recvcounts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_BYTE, 0, MPI_COMM_WORLD
        );
        MPI_Gatherv(
            local_threads.data(), local_tile_count, MPI_INT,
            rank == 0 ? tile_thread.data() : nullptr,
            rank == 0 ? t_counts.data() : nullptr,
            rank == 0 ? t_displs.data() : nullptr,
            MPI_INT, 0, MPI_COMM_WORLD
        );

        MPI_Reduce(&local_rays, &total_rays, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        double max_render_ms = 0.0;
        MPI_Reduce(&local_render_ms, &max_render_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        auto t_par_end = Clock::now();
        double parallel_wall_ms = elapsed_ms(t_par_start, t_par_end);
        tm.render_compute = std::min(parallel_wall_ms, max_render_ms);
        tm.comm_or_d2h = std::max(0.0, parallel_wall_ms - tm.render_compute);
        tm.total_compute = tm.scene_build + tm.bvh_build + parallel_wall_ms;
    } else {
        // Dynamic MPI Master-Worker with persistent OpenMP teams & pixel-balanced batches
        const int TAG_WORK = 1;
        const int TAG_RESULT = 2;
        const size_t max_tile_bytes = static_cast<size_t>(cfg.tile) * cfg.tile * 3;
        const int max_batch = std::max(1, num_tiles / std::max(1, world_size * 16));
        const int desired_batch = std::max(2, actual_threads * 2);
        const int batch_size = std::max(1, std::min(desired_batch, max_batch));
        const size_t max_pkt_bytes =
            sizeof(BatchResultHeader) + static_cast<size_t>(batch_size) * (sizeof(int) + max_tile_bytes);

        auto compute_batch_offsets = [&](int start_tile, int count, std::vector<size_t>& offs) -> size_t {
            offs.resize(static_cast<size_t>(count));
            size_t total_px_bytes = 0;
            for (int k = 0; k < count; ++k) {
                offs[k] = total_px_bytes;
                int x0, y0, x1, y1;
                get_tile_bounds(start_tile + k, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                total_px_bytes += static_cast<size_t>(x1 - x0) * (y1 - y0) * 3;
            }
            return total_px_bytes;
        };

        if (rank == 0) {
            std::vector<uint8_t> recv_buf(max_pkt_bytes);
            std::vector<size_t> tile_offs;
            std::atomic<int> next_tile{0};
            int active_workers = 0;
            double comm_ms = 0.0;
            uint64_t worker_rays = 0;

            auto make_next_batch = [&]() -> WorkBatch {
                int claimed = next_tile.fetch_add(batch_size, std::memory_order_relaxed);
                if (claimed >= num_tiles) return WorkBatch{-1, 0};
                int cnt = std::min(batch_size, num_tiles - claimed);
                return WorkBatch{claimed, cnt};
            };

            auto t_c0 = Clock::now();
            for (int w = 1; w < world_size; ++w) {
                WorkBatch wb = make_next_batch();
                MPI_Send(&wb, sizeof(WorkBatch), MPI_BYTE, w, TAG_WORK, MPI_COMM_WORLD);
                if (wb.count > 0) ++active_workers;
            }
            comm_ms += elapsed_ms(t_c0, Clock::now());

            auto unpack_and_dispatch = [&](int src) {
                BatchResultHeader hdr{};
                std::memcpy(&hdr, recv_buf.data(), sizeof(BatchResultHeader));
                const int* th_src = reinterpret_cast<const int*>(recv_buf.data() + sizeof(BatchResultHeader));
                const uint8_t* px_base =
                    recv_buf.data() + sizeof(BatchResultHeader) + static_cast<size_t>(hdr.count) * sizeof(int);

                compute_batch_offsets(hdr.start_tile, hdr.count, tile_offs);
                for (int k = 0; k < hdr.count; ++k) {
                    int tidx = hdr.start_tile + k;
                    int x0, y0, x1, y1;
                    get_tile_bounds(tidx, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                    int tw = x1 - x0;
                    const uint8_t* px_src = px_base + tile_offs[k];
                    for (int y = y0; y < y1; ++y) {
                        uint8_t* row_dst = &framebuffer[(static_cast<size_t>(y) * cfg.width + x0) * 3];
                        const uint8_t* row_src = px_src + static_cast<size_t>(y - y0) * tw * 3;
                        std::memcpy(row_dst, row_src, static_cast<size_t>(tw) * 3);
                    }
                    tile_rank[tidx] = src;
                    tile_thread[tidx] = th_src[k];
                }
                worker_rays += hdr.rays;

                WorkBatch next_wb = make_next_batch();
                MPI_Send(&next_wb, sizeof(WorkBatch), MPI_BYTE, src, TAG_WORK, MPI_COMM_WORLD);
                if (next_wb.count <= 0) --active_workers;
            };

            auto drain_nonblocking = [&]() {
                auto tc0 = Clock::now();
                while (active_workers > 0) {
                    int flag = 0;
                    MPI_Status status;
                    MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status);
                    if (!flag) break;
                    int bytes = 0;
                    MPI_Get_count(&status, MPI_BYTE, &bytes);
                    MPI_Recv(recv_buf.data(), bytes, MPI_BYTE, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    unpack_and_dispatch(status.MPI_SOURCE);
                }
                comm_ms += elapsed_ms(tc0, Clock::now());
            };

            uint64_t r0_rays = 0;
            // Single persistent OpenMP team on Rank 0: Thread 0 services MPI between scanlines;
            // Threads 1..T-1 render individual tiles continuously with zero per-tile barriers.
            #pragma omp parallel reduction(+:r0_rays)
            {
                int tid = omp_get_thread_num();
                if (tid == 0) {
                    while (true) {
                        drain_nonblocking();
                        int my_tile = next_tile.fetch_add(1, std::memory_order_relaxed);
                        if (my_tile >= num_tiles) break;
                        tile_rank[my_tile] = 0;
                        tile_thread[my_tile] = 0;
                        int x0, y0, x1, y1;
                        get_tile_bounds(my_tile, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                        for (int y = y0; y < y1; ++y) {
                            if (y > y0) drain_nonblocking();
                            render_tile_serial_pixels(view, cfg, x0, y, x1, y + 1, framebuffer.data(), cfg.width, 0, 0, r0_rays);
                        }
                    }
                } else {
                    while (true) {
                        int my_tile = next_tile.fetch_add(1, std::memory_order_relaxed);
                        if (my_tile >= num_tiles) break;
                        tile_rank[my_tile] = 0;
                        tile_thread[my_tile] = tid;
                        int x0, y0, x1, y1;
                        get_tile_bounds(my_tile, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                        render_tile_serial_pixels(view, cfg, x0, y0, x1, y1, framebuffer.data(), cfg.width, 0, 0, r0_rays);
                    }
                }
            }
            total_rays += r0_rays;

            // Other OpenMP threads have exited; main thread cleanly drains remaining workers
            auto tc_end0 = Clock::now();
            while (active_workers > 0) {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
                int bytes = 0;
                MPI_Get_count(&status, MPI_BYTE, &bytes);
                MPI_Recv(recv_buf.data(), bytes, MPI_BYTE, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                unpack_and_dispatch(status.MPI_SOURCE);
            }
            comm_ms += elapsed_ms(tc_end0, Clock::now());
            total_rays += worker_rays;

            auto t_par_end = Clock::now();
            double parallel_wall_ms = elapsed_ms(t_par_start, t_par_end);
            tm.comm_or_d2h = comm_ms;
            tm.render_compute = std::max(0.0, parallel_wall_ms - comm_ms);
            tm.total_compute = tm.scene_build + tm.bvh_build + parallel_wall_ms;
        } else {
            // Worker rank: single persistent OpenMP team collaborates at pixel granularity per batch
            configure_omp_schedule(cfg.schedule, 64);
            std::vector<uint8_t> pkt_buf(max_pkt_bytes);
            std::vector<size_t> tile_offs;

            struct BatchTileInfo {
                int x0, y0, tw, th, pix_start, pix_end;
            };
            std::vector<BatchTileInfo> tinfo(static_cast<size_t>(batch_size));
            WorkBatch cur_wb{};
            int total_batch_pixels = 0;
            size_t cur_px_bytes = 0;
            uint64_t batch_rays = 0;

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                while (true) {
                    if (tid == 0) {
                        MPI_Recv(&cur_wb, sizeof(WorkBatch), MPI_BYTE, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (cur_wb.count > 0) {
                            cur_px_bytes = compute_batch_offsets(cur_wb.start_tile, cur_wb.count, tile_offs);
                            int* th_dst = reinterpret_cast<int*>(pkt_buf.data() + sizeof(BatchResultHeader));
                            int acc_pix = 0;
                            for (int k = 0; k < cur_wb.count; ++k) {
                                int x0, y0, x1, y1;
                                get_tile_bounds(cur_wb.start_tile + k, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                                int tw = x1 - x0;
                                int th = y1 - y0;
                                tinfo[k] = BatchTileInfo{x0, y0, tw, th, acc_pix, acc_pix + tw * th};
                                acc_pix += tw * th;
                                th_dst[k] = 0;
                            }
                            total_batch_pixels = acc_pix;
                            batch_rays = 0;
                        }
                    }

                    #pragma omp barrier
                    if (cur_wb.count <= 0) break;

                    uint8_t* px_base =
                        pkt_buf.data() + sizeof(BatchResultHeader) + static_cast<size_t>(cur_wb.count) * sizeof(int);
                    uint64_t local_rays = 0;

                    #pragma omp for schedule(runtime)
                    for (int p = 0; p < total_batch_pixels; ++p) {
                        // Find tile k for pixel p (cur_wb.count is small, linear scan is ~2-8 steps)
                        int k = 0;
                        while (k + 1 < cur_wb.count && p >= tinfo[k + 1].pix_start) {
                            ++k;
                        }
                        const BatchTileInfo& ti = tinfo[k];
                        int local_p = p - ti.pix_start;
                        int ly = local_p / ti.tw;
                        int lx = local_p % ti.tw;
                        int x = ti.x0 + lx;
                        int y = ti.y0 + ly;
                        uint8_t* dst = px_base + tile_offs[k] + (static_cast<size_t>(ly) * ti.tw + lx) * 3;
                        render_pixel(view, x, y, cfg.width, cfg.height, cfg.spp, cfg.depth, cfg.seed, dst, local_rays);
                    }

                    #pragma omp atomic
                    batch_rays += local_rays;

                    #pragma omp barrier
                    if (tid == 0) {
                        BatchResultHeader hdr{cur_wb.start_tile, cur_wb.count, batch_rays};
                        std::memcpy(pkt_buf.data(), &hdr, sizeof(BatchResultHeader));
                        int send_bytes = static_cast<int>(
                            sizeof(BatchResultHeader) + static_cast<size_t>(cur_wb.count) * sizeof(int) + cur_px_bytes
                        );
                        MPI_Send(pkt_buf.data(), send_bytes, MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
                    }
                }
            }
        }
    }

    if (rank == 0) {
        auto t_io0 = Clock::now();
        if (!cfg.output.empty()) {
            ensure_parent_dir(cfg.output);
            if (!write_image_file(cfg.output, cfg.width, cfg.height, framebuffer.data())) {
                std::cerr << "Failed to write output image: " << cfg.output << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        if (!cfg.heatmap.empty()) {
            ensure_parent_dir(cfg.heatmap);
            write_heatmap_png(cfg.heatmap, cfg.width, cfg.height, cfg.tile, tile_rank, tile_thread, world_size, actual_threads);
        }
        auto t_io1 = Clock::now();
        tm.image_io = elapsed_ms(t_io0, t_io1);
        tm.total_wall = elapsed_ms(t_wall_start, t_io1);

        emit_json_telemetry(
            "hybrid",
            detect_cpu_name(),
            cfg,
            world_size,
            actual_threads,
            static_cast<int>(scene.prims.size()),
            scene.triangle_count,
            total_rays,
            tm
        );
    }

    MPI_Finalize();
    return 0;
}
