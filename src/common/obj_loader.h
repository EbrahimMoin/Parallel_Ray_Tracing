#pragma once

#include "primitive.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

inline bool load_obj_mesh(
    const std::string& path,
    int mat_id,
    float target_height,
    const Vec3& target_base_center,
    std::vector<Primitive>& out_prims,
    int& out_triangle_count
) {
    out_triangle_count = 0;
    std::ifstream in(path);
    if (!in.is_open()) return false;

    std::vector<Vec3> verts;
    std::vector<int> tri_indices;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string tok;
        iss >> tok;
        if (tok == "v") {
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (iss >> x >> y >> z) {
                verts.emplace_back(x, y, z);
            }
        } else if (tok == "f") {
            std::vector<int> poly;
            std::string vtok;
            while (iss >> vtok) {
                size_t slash = vtok.find('/');
                std::string idx_str = (slash == std::string::npos) ? vtok : vtok.substr(0, slash);
                if (idx_str.empty()) continue;
                int idx = std::stoi(idx_str);
                if (idx > 0) {
                    idx -= 1;
                } else if (idx < 0) {
                    idx = static_cast<int>(verts.size()) + idx;
                } else {
                    continue;
                }
                if (idx >= 0 && idx < static_cast<int>(verts.size())) {
                    poly.push_back(idx);
                }
            }
            for (size_t i = 1; i + 1 < poly.size(); ++i) {
                tri_indices.push_back(poly[0]);
                tri_indices.push_back(poly[i]);
                tri_indices.push_back(poly[i + 1]);
            }
        }
    }

    if (verts.empty() || tri_indices.size() < 3) return false;

    Vec3 bmin(RT_INF, RT_INF, RT_INF);
    Vec3 bmax(-RT_INF, -RT_INF, -RT_INF);
    for (const Vec3& v : verts) {
        bmin.x = rt_min(bmin.x, v.x);
        bmin.y = rt_min(bmin.y, v.y);
        bmin.z = rt_min(bmin.z, v.z);
        bmax.x = rt_max(bmax.x, v.x);
        bmax.y = rt_max(bmax.y, v.y);
        bmax.z = rt_max(bmax.z, v.z);
    }

    float height = rt_max(1e-5f, bmax.y - bmin.y);
    float scale = target_height / height;
    float cx = 0.5f * (bmin.x + bmax.x);
    float cz = 0.5f * (bmin.z + bmax.z);

    auto xform = [&](const Vec3& v) -> Vec3 {
        return Vec3(
            (v.x - cx) * scale + target_base_center.x,
            (v.y - bmin.y) * scale + target_base_center.y,
            (v.z - cz) * scale + target_base_center.z
        );
    };

    for (size_t i = 0; i + 2 < tri_indices.size(); i += 3) {
        Vec3 a = xform(verts[tri_indices[i]]);
        Vec3 b = xform(verts[tri_indices[i + 1]]);
        Vec3 c = xform(verts[tri_indices[i + 2]]);
        if (cross(b - a, c - a).length_squared() < 1e-14f) continue;
        out_prims.push_back(Primitive::make_triangle(a, b, c, mat_id));
        ++out_triangle_count;
    }

    return out_triangle_count > 0;
}
