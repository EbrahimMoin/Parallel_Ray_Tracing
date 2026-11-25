#include "cli_and_io.h"
#include "renderer.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <mpi.h>
#include <vector>

static double elapsed_ms(
    const std::chrono::steady_clock::time_point& a,
    const std::chrono::steady_clock::time_point& b
) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct TilePacketHeader {
    int      tile_idx;
    int      owner_thread;
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

static void render_tile_buffer(
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    RenderConfig cfg;
    if (!parse_cli(argc, argv, cfg)) {
        MPI_Finalize();
        return (rank == 0) ? 1 : 0;
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
        // Direct single-rank fast path (no MPI message loop)
        for (int tidx = 0; tidx < num_tiles; ++tidx) {
            int x0, y0, x1, y1;
            get_tile_bounds(tidx, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
            render_tile_buffer(view, cfg, x0, y0, x1, y1, framebuffer.data(), cfg.width, 0, 0, total_rays);
        }
        auto t_par_end = Clock::now();
        tm.render_compute = elapsed_ms(t_par_start, t_par_end);
        tm.comm_or_d2h = 0.0;
        tm.total_compute = tm.scene_build + tm.bvh_build + tm.render_compute;
    } else if (cfg.mpi_mode == "static") {
        // Static contiguous tile-row decomposition + collective MPI_Gatherv
        int my_ty0 = 0, my_ty1 = 0, my_y0 = 0, my_y1 = 0;
        rank_tile_rows(rank, world_size, tiles_y, cfg.tile, cfg.height, my_ty0, my_ty1, my_y0, my_y1);

        int local_rows = my_y1 - my_y0;
        int local_bytes = local_rows * cfg.width * 3;
        std::vector<uint8_t> local_buf(static_cast<size_t>(local_bytes), 0);

        uint64_t local_rays = 0;
        auto t_loc0 = Clock::now();
        for (int ty = my_ty0; ty < my_ty1; ++ty) {
            for (int tx = 0; tx < tiles_x; ++tx) {
                int tidx = ty * tiles_x + tx;
                int x0, y0, x1, y1;
                get_tile_bounds(tidx, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                render_tile_buffer(view, cfg, x0, y0, x1, y1, local_buf.data(), cfg.width, 0, my_y0, local_rays);
            }
        }
        auto t_loc1 = Clock::now();
        double local_render_ms = elapsed_ms(t_loc0, t_loc1);

        std::vector<int> recvcounts;
        std::vector<int> displs;
        if (rank == 0) {
            recvcounts.resize(world_size);
            displs.resize(world_size);
            for (int r = 0; r < world_size; ++r) {
                int r_ty0, r_ty1, r_y0, r_y1;
                rank_tile_rows(r, world_size, tiles_y, cfg.tile, cfg.height, r_ty0, r_ty1, r_y0, r_y1);
                recvcounts[r] = (r_y1 - r_y0) * cfg.width * 3;
                displs[r] = r_y0 * cfg.width * 3;
                for (int ty = r_ty0; ty < r_ty1; ++ty) {
                    for (int tx = 0; tx < tiles_x; ++tx) {
                        tile_rank[ty * tiles_x + tx] = r;
                    }
                }
            }
        }

        MPI_Gatherv(
            local_buf.data(),
            local_bytes,
            MPI_BYTE,
            rank == 0 ? framebuffer.data() : nullptr,
            rank == 0 ? recvcounts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
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
        // Dynamic self-participating Master-Worker queue via MPI_Iprobe
        const int TAG_WORK = 1;
        const int TAG_RESULT = 2;
        const size_t max_tile_bytes = static_cast<size_t>(cfg.tile) * cfg.tile * 3;

        if (rank == 0) {
            std::vector<uint8_t> recv_buf(sizeof(TilePacketHeader) + max_tile_bytes);
            int next_tile = 0;
            int active_workers = 0;
            double comm_ms = 0.0;

            auto t_c0 = Clock::now();
            for (int w = 1; w < world_size; ++w) {
                int msg = (next_tile < num_tiles) ? next_tile++ : -1;
                MPI_Send(&msg, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
                if (msg >= 0) ++active_workers;
            }
            comm_ms += elapsed_ms(t_c0, Clock::now());

            auto unpack_and_dispatch = [&](int src, int msg_bytes) {
                TilePacketHeader hdr{};
                std::memcpy(&hdr, recv_buf.data(), sizeof(TilePacketHeader));
                const uint8_t* px_src = recv_buf.data() + sizeof(TilePacketHeader);

                int x0, y0, x1, y1;
                get_tile_bounds(hdr.tile_idx, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                int tw = x1 - x0;
                int th = y1 - y0;
                (void)msg_bytes;

                for (int y = y0; y < y1; ++y) {
                    uint8_t* row_dst = &framebuffer[(static_cast<size_t>(y) * cfg.width + x0) * 3];
                    const uint8_t* row_src = px_src + static_cast<size_t>(y - y0) * tw * 3;
                    std::memcpy(row_dst, row_src, static_cast<size_t>(tw) * 3);
                }
                (void)th;

                tile_rank[hdr.tile_idx] = src;
                tile_thread[hdr.tile_idx] = hdr.owner_thread;
                total_rays += hdr.rays;

                int next_msg = (next_tile < num_tiles) ? next_tile++ : -1;
                MPI_Send(&next_msg, 1, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);
                if (next_msg < 0) --active_workers;
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
                    unpack_and_dispatch(status.MPI_SOURCE, bytes);
                }
                comm_ms += elapsed_ms(tc0, Clock::now());
            };

            while (next_tile < num_tiles) {
                drain_nonblocking();
                if (next_tile >= num_tiles) break;

                int my_tile = next_tile++;
                tile_rank[my_tile] = 0;
                tile_thread[my_tile] = 0;
                int x0, y0, x1, y1;
                get_tile_bounds(my_tile, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                // Service workers between scanlines so heavy tiles on Rank 0 never stall workers
                for (int y = y0; y < y1; ++y) {
                    if (y > y0) drain_nonblocking();
                    render_tile_buffer(view, cfg, x0, y, x1, y + 1, framebuffer.data(), cfg.width, 0, 0, total_rays);
                }
            }

            auto tc_end0 = Clock::now();
            while (active_workers > 0) {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
                int bytes = 0;
                MPI_Get_count(&status, MPI_BYTE, &bytes);
                MPI_Recv(recv_buf.data(), bytes, MPI_BYTE, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                unpack_and_dispatch(status.MPI_SOURCE, bytes);
            }
            comm_ms += elapsed_ms(tc_end0, Clock::now());

            auto t_par_end = Clock::now();
            double parallel_wall_ms = elapsed_ms(t_par_start, t_par_end);
            tm.comm_or_d2h = comm_ms;
            tm.render_compute = std::max(0.0, parallel_wall_ms - comm_ms);
            tm.total_compute = tm.scene_build + tm.bvh_build + parallel_wall_ms;
        } else {
            // Worker rank: allocate only a single tile buffer
            std::vector<uint8_t> pkt_buf(sizeof(TilePacketHeader) + max_tile_bytes);
            while (true) {
                int tidx = -1;
                MPI_Recv(&tidx, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (tidx < 0) break;

                int x0, y0, x1, y1;
                get_tile_bounds(tidx, tiles_x, cfg.tile, cfg.width, cfg.height, x0, y0, x1, y1);
                int tw = x1 - x0;
                int th = y1 - y0;

                uint64_t tile_rays = 0;
                uint8_t* px_dst = pkt_buf.data() + sizeof(TilePacketHeader);
                render_tile_buffer(view, cfg, x0, y0, x1, y1, px_dst, tw, x0, y0, tile_rays);

                TilePacketHeader hdr{tidx, 0, tile_rays};
                std::memcpy(pkt_buf.data(), &hdr, sizeof(TilePacketHeader));
                int send_bytes = static_cast<int>(sizeof(TilePacketHeader) + static_cast<size_t>(tw) * th * 3);
                MPI_Send(pkt_buf.data(), send_bytes, MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
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
            write_heatmap_png(cfg.heatmap, cfg.width, cfg.height, cfg.tile, tile_rank, tile_thread, world_size, 1);
        }
        auto t_io1 = Clock::now();
        tm.image_io = elapsed_ms(t_io0, t_io1);
        tm.total_wall = elapsed_ms(t_wall_start, t_io1);

        emit_json_telemetry(
            "mpi",
            detect_cpu_name(),
            cfg,
            world_size,
            1,
            static_cast<int>(scene.prims.size()),
            scene.triangle_count,
            total_rays,
            tm
        );
    }

    MPI_Finalize();
    return 0;
}
