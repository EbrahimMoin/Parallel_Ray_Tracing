#include "cli_and_io.h"
#include "renderer.h"
#include <chrono>
#include <omp.h>
#include <vector>

static double elapsed_ms(
    const std::chrono::steady_clock::time_point& a,
    const std::chrono::steady_clock::time_point& b
) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void configure_omp_schedule(const std::string& sched) {
    if (sched == "static") {
        omp_set_schedule(omp_sched_static, 0);
    } else if (sched == "guided") {
        omp_set_schedule(omp_sched_guided, 1);
    } else {
        omp_set_schedule(omp_sched_dynamic, 1);
    }
}

int main(int argc, char** argv) {
    RenderConfig cfg;
    if (!parse_cli(argc, argv, cfg)) {
        return 1;
    }

    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }
    configure_omp_schedule(cfg.schedule);

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

    SceneView view = scene.view();
    std::vector<uint8_t> framebuffer(static_cast<size_t>(cfg.width) * cfg.height * 3, 0);

    int tiles_x = (cfg.width + cfg.tile - 1) / cfg.tile;
    int tiles_y = (cfg.height + cfg.tile - 1) / cfg.tile;
    int num_tiles = tiles_x * tiles_y;
    std::vector<int> tile_rank(num_tiles, 0);
    std::vector<int> tile_thread(num_tiles, 0);

    uint64_t total_rays = 0;
    auto t3 = Clock::now();

    #pragma omp parallel for schedule(runtime) reduction(+:total_rays)
    for (int tidx = 0; tidx < num_tiles; ++tidx) {
        int tid = omp_get_thread_num();
        tile_thread[tidx] = tid;

        int ty = tidx / tiles_x;
        int tx = tidx % tiles_x;
        int y0 = ty * cfg.tile;
        int y1 = rt_min(y0 + cfg.tile, cfg.height);
        int x0 = tx * cfg.tile;
        int x1 = rt_min(x0 + cfg.tile, cfg.width);

        uint64_t local_rays = 0;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                uint8_t* px = &framebuffer[(static_cast<size_t>(y) * cfg.width + x) * 3];
                render_pixel(view, x, y, cfg.width, cfg.height, cfg.spp, cfg.depth, cfg.seed, px, local_rays);
            }
        }
        total_rays += local_rays;
    }

    auto t4 = Clock::now();
    tm.render_compute = elapsed_ms(t3, t4);
    tm.total_compute = tm.scene_build + tm.bvh_build + tm.render_compute;

    auto t5 = Clock::now();
    if (!cfg.output.empty()) {
        ensure_parent_dir(cfg.output);
        if (!write_image_file(cfg.output, cfg.width, cfg.height, framebuffer.data())) {
            std::cerr << "Failed to write output image: " << cfg.output << "\n";
            return 1;
        }
    }
    if (!cfg.heatmap.empty()) {
        ensure_parent_dir(cfg.heatmap);
        write_heatmap_png(cfg.heatmap, cfg.width, cfg.height, cfg.tile, tile_rank, tile_thread, 1, actual_threads);
    }
    auto t6 = Clock::now();
    tm.image_io = elapsed_ms(t5, t6);
    tm.total_wall = elapsed_ms(t_wall_start, t6);

    emit_json_telemetry(
        "omp",
        detect_cpu_name(),
        cfg,
        1,
        actual_threads,
        static_cast<int>(scene.prims.size()),
        scene.triangle_count,
        total_rays,
        tm
    );
    return 0;
}
