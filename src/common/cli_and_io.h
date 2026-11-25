#pragma once

#include "png_writer.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

struct RenderConfig {
    std::string scene = "large_world";
    std::string obj_path = "assets/sample_mesh.obj";
    int width = 1280;
    int height = 720;
    int spp = 8;
    int depth = 8;
    int objects = 500;
    uint32_t seed = 42u;
    int tile = 32;
    int threads = 0; // 0 = auto / runtime default
    std::string schedule = "dynamic";
    std::string mpi_mode = "dynamic";
    int block = 16;
    std::string output = "results/renders/output.png";
    std::string heatmap = "";
    std::string stats_json = "";
};

struct TimingStats {
    double scene_build = 0.0;
    double bvh_build = 0.0;
    double h2d_transfer = 0.0;
    double render_compute = 0.0;
    double comm_or_d2h = 0.0;
    double image_io = 0.0;
    double total_compute = 0.0;
    double total_wall = 0.0;
};

inline void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --scene <large_world|studio_glass|mesh_showcase|cyberpunk_city|cyber_cathedral|orbital_megastructure>\n"
              << "  --obj <path.obj>\n"
              << "  --width <int>        --height <int>\n"
              << "  --spp <int>          --depth <int>\n"
              << "  --objects <int>      --seed <uint32>\n"
              << "  --tile <int>         --threads <int>\n"
              << "  --schedule <static|dynamic|guided>\n"
              << "  --mpi-mode <static|dynamic>\n"
              << "  --block <int>\n"
              << "  --output <path.png>  --heatmap <path.png>\n"
              << "  --stats-json <path.json>\n";
}

inline bool parse_cli(int argc, char** argv, RenderConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--scene") {
            const char* v = need_val("--scene"); if (!v) return false;
            cfg.scene = v;
        } else if (arg == "--obj") {
            const char* v = need_val("--obj"); if (!v) return false;
            cfg.obj_path = v;
        } else if (arg == "--width") {
            const char* v = need_val("--width"); if (!v) return false;
            cfg.width = std::max(16, std::atoi(v));
        } else if (arg == "--height") {
            const char* v = need_val("--height"); if (!v) return false;
            cfg.height = std::max(16, std::atoi(v));
        } else if (arg == "--spp") {
            const char* v = need_val("--spp"); if (!v) return false;
            cfg.spp = std::max(1, std::atoi(v));
        } else if (arg == "--depth") {
            const char* v = need_val("--depth"); if (!v) return false;
            cfg.depth = std::max(1, std::atoi(v));
        } else if (arg == "--objects") {
            const char* v = need_val("--objects"); if (!v) return false;
            cfg.objects = std::max(0, std::atoi(v));
        } else if (arg == "--seed") {
            const char* v = need_val("--seed"); if (!v) return false;
            cfg.seed = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        } else if (arg == "--tile") {
            const char* v = need_val("--tile"); if (!v) return false;
            cfg.tile = std::max(4, std::atoi(v));
        } else if (arg == "--threads") {
            const char* v = need_val("--threads"); if (!v) return false;
            cfg.threads = std::max(1, std::atoi(v));
        } else if (arg == "--schedule") {
            const char* v = need_val("--schedule"); if (!v) return false;
            cfg.schedule = v;
        } else if (arg == "--mpi-mode") {
            const char* v = need_val("--mpi-mode"); if (!v) return false;
            cfg.mpi_mode = v;
        } else if (arg == "--block") {
            const char* v = need_val("--block"); if (!v) return false;
            cfg.block = std::max(4, std::atoi(v));
        } else if (arg == "--output") {
            const char* v = need_val("--output"); if (!v) return false;
            cfg.output = v;
        } else if (arg == "--heatmap") {
            const char* v = need_val("--heatmap"); if (!v) return false;
            cfg.heatmap = v;
        } else if (arg == "--stats-json") {
            const char* v = need_val("--stats-json"); if (!v) return false;
            cfg.stats_json = v;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

inline void ensure_parent_dir(const std::string& path) {
    if (path.empty()) return;
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
}

inline std::string detect_cpu_name() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string name = line.substr(colon + 1);
                size_t first = name.find_first_not_of(" \t");
                if (first != std::string::npos) return name.substr(first);
            }
        }
    }
    return "CPU";
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

inline void emit_json_telemetry(
    const std::string& backend,
    const std::string& device_name,
    const RenderConfig& cfg,
    int ranks,
    int threads,
    int prim_count,
    int tri_count,
    uint64_t total_rays,
    const TimingStats& tm
) {
    double compute_sec = tm.total_compute > 0.0 ? (tm.total_compute / 1000.0) : 1e-9;
    double mrays_per_sec = (static_cast<double>(total_rays) / 1e6) / compute_sec;

    std::ostringstream js;
    js << std::fixed << std::setprecision(3);
    js << "{"
       << "\"backend\":\"" << json_escape(backend) << "\","
       << "\"device_name\":\"" << json_escape(device_name) << "\","
       << "\"scene\":\"" << json_escape(cfg.scene) << "\","
       << "\"width\":" << cfg.width << ","
       << "\"height\":" << cfg.height << ","
       << "\"spp\":" << cfg.spp << ","
       << "\"depth\":" << cfg.depth << ","
       << "\"objects\":" << cfg.objects << ","
       << "\"primitives\":" << prim_count << ","
       << "\"triangles\":" << tri_count << ","
       << "\"seed\":" << cfg.seed << ","
       << "\"tile\":" << cfg.tile << ","
       << "\"ranks\":" << ranks << ","
       << "\"threads\":" << threads << ","
       << "\"schedule\":\"" << json_escape(cfg.schedule) << "\","
       << "\"mpi_mode\":\"" << json_escape(cfg.mpi_mode) << "\","
       << "\"block\":" << cfg.block << ","
       << "\"total_rays\":" << total_rays << ","
       << "\"mrays_per_sec\":" << mrays_per_sec << ","
       << "\"output_image\":\"" << json_escape(cfg.output) << "\","
       << "\"heatmap_image\":\"" << json_escape(cfg.heatmap) << "\","
       << "\"timings_ms\":{"
       << "\"scene_build\":" << tm.scene_build << ","
       << "\"bvh_build\":" << tm.bvh_build << ","
       << "\"h2d_transfer\":" << tm.h2d_transfer << ","
       << "\"render_compute\":" << tm.render_compute << ","
       << "\"comm_or_d2h\":" << tm.comm_or_d2h << ","
       << "\"image_io\":" << tm.image_io << ","
       << "\"total_compute\":" << tm.total_compute << ","
       << "\"total_wall\":" << tm.total_wall
       << "}"
       << "}";

    std::string payload = js.str();
    std::cout << "JSON_RESULT:" << payload << "\n" << std::flush;

    if (!cfg.stats_json.empty()) {
        ensure_parent_dir(cfg.stats_json);
        std::ofstream f(cfg.stats_json);
        if (f.is_open()) {
            f << payload << "\n";
        }
    }
}
