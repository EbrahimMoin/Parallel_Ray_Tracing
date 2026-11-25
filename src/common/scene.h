#pragma once

#include "bvh.h"
#include "material.h"
#include "obj_loader.h"
#include <string>
#include <vector>

struct Camera {
    Vec3  origin;
    Vec3  lower_left_corner;
    Vec3  horizontal;
    Vec3  vertical;
    Vec3  u, v, w;
    float lens_radius;

    RT_HD static Camera create(
        const Vec3& lookfrom,
        const Vec3& lookat,
        const Vec3& vup,
        float vfov_deg,
        float aspect_ratio,
        float aperture,
        float focus_dist
    ) {
        Camera cam;
        float theta = vfov_deg * (RT_PI / 180.0f);
        float h = tanf(theta * 0.5f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        cam.w = normalize(lookfrom - lookat);
        cam.u = normalize(cross(vup, cam.w));
        cam.v = cross(cam.w, cam.u);

        cam.origin = lookfrom;
        cam.horizontal = focus_dist * viewport_width * cam.u;
        cam.vertical = focus_dist * viewport_height * cam.v;
        cam.lower_left_corner =
            cam.origin - 0.5f * cam.horizontal - 0.5f * cam.vertical - focus_dist * cam.w;
        cam.lens_radius = aperture * 0.5f;
        return cam;
    }

    RT_HD Ray get_ray(float s, float t, Rng& rng) const {
        if (lens_radius > 0.0f) {
            Vec3 rd = lens_radius * rng.random_in_unit_disk();
            Vec3 offset = u * rd.x + v * rd.y;
            return Ray(origin + offset,
                       normalize(lower_left_corner + s * horizontal + t * vertical - origin - offset));
        }
        return Ray(origin,
                   normalize(lower_left_corner + s * horizontal + t * vertical - origin));
    }
};

struct SceneView {
    const BVHNode*   nodes;
    int              node_count;
    const Primitive* prims;
    int              prim_count;
    const Material*  mats;
    int              mat_count;
    Camera           cam;
    Vec3             sky_top;
    Vec3             sky_bottom;
    Vec3             sun_dir;
    Vec3             sun_color;
    float            sun_intensity;
    Vec3             fog_color;
    float            fog_density;
};

struct SceneData {
    std::vector<Primitive> prims;
    std::vector<Material>  mats;
    std::vector<BVHNode>   nodes;
    Camera cam;
    Vec3   sky_top;
    Vec3   sky_bottom;
    Vec3   sun_dir;
    Vec3   sun_color;
    float  sun_intensity = 0.0f;
    Vec3   fog_color = Vec3(0.0f);
    float  fog_density = 0.0f;
    int    sphere_count = 0;
    int    triangle_count = 0;

    SceneView view() const {
        SceneView v;
        v.nodes = nodes.data();
        v.node_count = static_cast<int>(nodes.size());
        v.prims = prims.data();
        v.prim_count = static_cast<int>(prims.size());
        v.mats = mats.data();
        v.mat_count = static_cast<int>(mats.size());
        v.cam = cam;
        v.sky_top = sky_top;
        v.sky_bottom = sky_bottom;
        v.sun_dir = sun_dir;
        v.sun_color = sun_color;
        v.sun_intensity = sun_intensity;
        v.fog_color = fog_color;
        v.fog_density = fog_density;
        return v;
    }
};

inline void add_procedural_crystal_fallback(
    int mat_id,
    float scale,
    const Vec3& base_center,
    std::vector<Primitive>& prims,
    int& tri_count
) {
    const Vec3 v[6] = {
        base_center + Vec3(0.0f, 2.2f * scale, 0.0f),
        base_center + Vec3(1.1f * scale, 1.0f * scale, 0.0f),
        base_center + Vec3(0.0f, 1.0f * scale, 1.1f * scale),
        base_center + Vec3(-1.1f * scale, 1.0f * scale, 0.0f),
        base_center + Vec3(0.0f, 1.0f * scale, -1.1f * scale),
        base_center + Vec3(0.0f, 0.0f, 0.0f)
    };
    const int f[8][3] = {
        {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1},
        {5, 2, 1}, {5, 3, 2}, {5, 4, 3}, {5, 1, 4}
    };
    for (int i = 0; i < 8; ++i) {
        prims.push_back(Primitive::make_triangle(v[f[i][0]], v[f[i][1]], v[f[i][2]], mat_id));
        ++tri_count;
    }
}

inline void build_scene(
    const std::string& preset,
    int objects,
    uint32_t seed,
    int width,
    int height,
    const std::string& obj_path,
    SceneData& out
) {
    out.prims.clear();
    out.mats.clear();
    out.nodes.clear();
    out.fog_density = 0.0f;
    out.fog_color = Vec3(0.0f);
    out.sphere_count = 0;
    out.triangle_count = 0;

    Rng rng(seed ^ 0xA511E9B3u);
    float aspect = static_cast<float>(width) / static_cast<float>(rt_max(1, height));

    auto add_mat = [&](const Material& m) -> int {
        int id = static_cast<int>(out.mats.size());
        out.mats.push_back(m);
        return id;
    };
    auto add_sphere = [&](const Vec3& c, float r, int mat_id) {
        out.prims.push_back(Primitive::make_sphere(c, r, mat_id));
        ++out.sphere_count;
    };
    auto add_tri = [&](const Vec3& a, const Vec3& b, const Vec3& c, int mat_id) {
        if (cross(b - a, c - a).length_squared() < 1e-14f) return;
        out.prims.push_back(Primitive::make_triangle(a, b, c, mat_id));
        ++out.triangle_count;
    };
    auto add_box = [&](const Vec3& mn, const Vec3& mx, int mat_id) {
        Vec3 p000(mn.x, mn.y, mn.z), p100(mx.x, mn.y, mn.z);
        Vec3 p110(mx.x, mx.y, mn.z), p010(mn.x, mx.y, mn.z);
        Vec3 p001(mn.x, mn.y, mx.z), p101(mx.x, mn.y, mx.z);
        Vec3 p111(mx.x, mx.y, mx.z), p011(mn.x, mx.y, mx.z);
        // Front (+Z) & Back (-Z)
        add_tri(p001, p101, p111, mat_id); add_tri(p001, p111, p011, mat_id);
        add_tri(p100, p000, p010, mat_id); add_tri(p100, p010, p110, mat_id);
        // Right (+X) & Left (-X)
        add_tri(p101, p100, p110, mat_id); add_tri(p101, p110, p111, mat_id);
        add_tri(p000, p001, p011, mat_id); add_tri(p000, p011, p010, mat_id);
        // Top (+Y) & Bottom (-Y)
        add_tri(p011, p111, p110, mat_id); add_tri(p011, p110, p010, mat_id);
        add_tri(p000, p100, p101, mat_id); add_tri(p000, p101, p001, mat_id);
    };
    auto add_crystal_shard = [&](const Vec3& c, float rx, float ry, float rz, float yaw, int mat_id) {
        float cy = cosf(yaw), sy = sinf(yaw);
        auto rot = [&](float lx, float ly, float lz) -> Vec3 {
            return c + Vec3(lx * cy - lz * sy, ly, lx * sy + lz * cy);
        };
        Vec3 top = rot(0.0f, ry, 0.0f);
        Vec3 bot = rot(0.0f, -ry, 0.0f);
        Vec3 eq[4] = {
            rot( rx, 0.0f,  0.0f),
            rot( 0.0f, 0.0f,  rz),
            rot(-rx, 0.0f,  0.0f),
            rot( 0.0f, 0.0f, -rz)
        };
        for (int k = 0; k < 4; ++k) {
            int kn = (k + 1) & 3;
            add_tri(top, eq[kn], eq[k], mat_id);
            add_tri(bot, eq[k], eq[kn], mat_id);
        }
    };

    if (preset == "studio_glass") {
        // High workload imbalance: dark sky (1 ray) vs dense glass/mirror cluster (many bounces)
        out.sky_top = Vec3(0.03f, 0.05f, 0.11f);
        out.sky_bottom = Vec3(0.11f, 0.15f, 0.24f);
        out.sun_dir = normalize(Vec3(0.4f, 0.8f, 0.3f));
        out.sun_color = Vec3(1.0f, 0.92f, 0.82f);
        out.sun_intensity = 1.5f;

        Vec3 lookfrom(0.0f, 2.3f, 9.2f);
        Vec3 lookat(0.0f, 1.0f, 0.0f);
        out.cam = Camera::create(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 32.0f, aspect, 0.0f, 9.2f);

        int m_floor = add_mat(Material::checker(Vec3(0.09f, 0.10f, 0.12f), Vec3(0.18f, 0.20f, 0.24f), 1.25f, 0.35f));
        int m_glass_clear = add_mat(Material::glass(1.52f, Vec3(0.98f, 0.99f, 1.0f)));
        int m_glass_cyan  = add_mat(Material::glass(1.55f, Vec3(0.78f, 0.96f, 1.0f)));
        int m_glass_amber = add_mat(Material::glass(1.50f, Vec3(1.0f, 0.88f, 0.72f)));
        int m_chrome      = add_mat(Material::metal(Vec3(0.92f, 0.94f, 0.96f), 0.01f));
        int m_gold        = add_mat(Material::metal(Vec3(0.95f, 0.76f, 0.42f), 0.04f));
        int m_light_warm  = add_mat(Material::emissive(Vec3(12.0f, 10.5f, 8.5f)));
        int m_light_cool  = add_mat(Material::emissive(Vec3(7.0f, 10.0f, 14.0f)));

        add_sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, m_floor);

        // Overhead and rim emissive lights
        add_sphere(Vec3(-2.6f, 5.2f, 1.8f), 1.05f, m_light_warm);
        add_sphere(Vec3( 2.8f, 4.9f, -1.5f), 0.95f, m_light_cool);
        add_sphere(Vec3( 0.0f, 6.2f, -3.5f), 1.20f, m_light_warm);

        // Hero cluster in the center
        add_sphere(Vec3( 0.0f, 1.25f, 0.0f), 1.25f, m_glass_clear);
        add_sphere(Vec3(-1.9f, 0.95f, -0.6f), 0.95f, m_chrome);
        add_sphere(Vec3( 1.9f, 0.95f, -0.4f), 0.95f, m_gold);

        int count = objects > 0 ? objects : 120;
        for (int i = 0; i < count; ++i) {
            float angle = 2.399963f * static_cast<float>(i) + rng.next_float(-0.15f, 0.15f);
            float rad_ring = 1.6f + 0.22f * sqrtf(static_cast<float>(i + 1));
            float r = rng.next_float(0.16f, 0.42f);
            float x = cosf(angle) * rad_ring;
            float z = sinf(angle) * rad_ring * 0.75f;
            float y = r + (i % 5 == 0 ? rng.next_float(0.4f, 1.6f) : 0.0f);

            // Keep central hero corridor clear
            if ((x * x + z * z) < 2.2f * 2.2f && y < 2.3f) {
                x += (x >= 0.0f ? 1.8f : -1.8f);
            }

            int pick = i % 6;
            int mat_id = m_glass_clear;
            if (pick == 1) mat_id = m_glass_cyan;
            else if (pick == 2) mat_id = m_chrome;
            else if (pick == 3) mat_id = m_glass_amber;
            else if (pick == 4) mat_id = m_gold;
            else if (pick == 5) {
                Vec3 c(rng.next_float(0.15f, 0.85f), rng.next_float(0.15f, 0.85f), rng.next_float(0.25f, 0.95f));
                mat_id = add_mat(Material::diffuse(c));
            }
            add_sphere(Vec3(x, y, z), r, mat_id);
        }
    } else if (preset == "mesh_showcase") {
        out.sky_top = Vec3(0.08f, 0.14f, 0.26f);
        out.sky_bottom = Vec3(0.28f, 0.36f, 0.48f);
        out.sun_dir = normalize(Vec3(0.55f, 0.75f, 0.35f));
        out.sun_color = Vec3(1.0f, 0.94f, 0.85f);
        out.sun_intensity = 2.5f;

        Vec3 lookfrom(3.6f, 2.5f, 5.4f);
        Vec3 lookat(0.0f, 1.15f, 0.0f);
        out.cam = Camera::create(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 34.0f, aspect, 0.0f, 6.2f);

        int m_floor = add_mat(Material::checker(Vec3(0.14f, 0.16f, 0.19f), Vec3(0.32f, 0.35f, 0.40f), 1.2f, 0.22f));
        int m_mesh  = add_mat(Material::metal(Vec3(0.92f, 0.72f, 0.38f), 0.08f));
        int m_glass = add_mat(Material::glass(1.52f, Vec3(0.95f, 0.99f, 1.0f)));
        int m_silver = add_mat(Material::metal(Vec3(0.88f, 0.90f, 0.94f), 0.03f));
        int m_light1 = add_mat(Material::emissive(Vec3(11.0f, 10.0f, 8.5f)));
        int m_light2 = add_mat(Material::emissive(Vec3(6.5f, 9.0f, 13.0f)));

        add_sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, m_floor);
        add_sphere(Vec3(-3.2f, 4.8f, 2.5f), 0.95f, m_light1);
        add_sphere(Vec3( 3.5f, 4.2f, -2.5f), 0.85f, m_light2);

        std::string resolved_obj = obj_path.empty() ? "assets/sample_mesh.obj" : obj_path;
        int mesh_tris = 0;
        if (!load_obj_mesh(resolved_obj, m_mesh, 2.5f, Vec3(0.0f, 0.02f, 0.0f), out.prims, mesh_tris)) {
            add_procedural_crystal_fallback(m_mesh, 1.1f, Vec3(0.0f, 0.02f, 0.0f), out.prims, mesh_tris);
        }
        out.triangle_count += mesh_tris;

        // Flanking hero glass/metal spheres and accent orbs
        add_sphere(Vec3(-2.3f, 0.75f, 1.0f), 0.75f, m_glass);
        add_sphere(Vec3( 2.3f, 0.75f, -0.6f), 0.75f, m_silver);

        int extra = objects > 0 ? objects : 80;
        for (int i = 0; i < extra; ++i) {
            float ang = 2.399963f * static_cast<float>(i);
            float dist = 2.8f + 0.35f * sqrtf(static_cast<float>(i + 1));
            float r = rng.next_float(0.14f, 0.32f);
            float x = cosf(ang) * dist;
            float z = sinf(ang) * dist;
            int mat_id;
            if (i % 3 == 0) mat_id = m_glass;
            else if (i % 3 == 1) mat_id = m_silver;
            else {
                Vec3 col(rng.next_float(0.2f, 0.85f), rng.next_float(0.2f, 0.85f), rng.next_float(0.25f, 0.90f));
                mat_id = add_mat(Material::diffuse(col));
            }
            add_sphere(Vec3(x, r, z), r, mat_id);
        }
    } else if (preset == "cyberpunk_city") {
        // Rainy Cyberpunk Boulevard with Widebody Sports Car, Elevated Monorail,
        // Neon Glyph Billboards, Skyscraper Facades, and Umbrella Pedestrians
        out.sky_top = Vec3(0.012f, 0.025f, 0.052f);
        out.sky_bottom = Vec3(0.055f, 0.115f, 0.185f);
        out.sun_dir = normalize(Vec3(-0.18f, 0.42f, -0.89f));
        out.sun_color = Vec3(0.55f, 0.85f, 1.0f);
        out.sun_intensity = 1.6f;
        out.fog_color = Vec3(0.032f, 0.072f, 0.115f);
        out.fog_density = 0.016f;

        Vec3 lookfrom(-0.35f, 0.96f, 8.65f);
        Vec3 lookat(0.05f, 1.38f, -4.0f);
        out.cam = Camera::create(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 40.0f, aspect, 0.0f, 12.0f);

        // Materials
        int m_asphalt    = add_mat(Material::wet_asphalt(Vec3(0.026f, 0.030f, 0.040f), 0.56f));
        int m_sidewalk   = add_mat(Material::wet_asphalt(Vec3(0.055f, 0.062f, 0.075f), 0.30f));
        int m_car_body   = add_mat(Material::car_paint(Vec3(0.36f, 0.44f, 0.54f), 0.09f));
        int m_car_carbon = add_mat(Material::metal(Vec3(0.07f, 0.08f, 0.10f), 0.22f));
        int m_car_chrome = add_mat(Material::metal(Vec3(0.88f, 0.92f, 0.97f), 0.03f));
        int m_car_glass  = add_mat(Material::glass(1.52f, Vec3(0.38f, 0.52f, 0.62f)));
        int m_tire       = add_mat(Material::diffuse(Vec3(0.035f, 0.038f, 0.044f)));
        int m_headlight  = add_mat(Material::emissive(Vec3(11.5f, 13.0f, 15.0f)));
        int m_taillight  = add_mat(Material::emissive(Vec3(12.0f, 0.9f, 0.6f)));
        int m_under_cyn  = add_mat(Material::emissive(Vec3(0.6f, 6.2f, 8.2f)));
        int m_amber_led  = add_mat(Material::emissive(Vec3(10.0f, 4.2f, 0.4f)));

        int m_steel      = add_mat(Material::metal(Vec3(0.11f, 0.13f, 0.17f), 0.30f));
        int m_concrete   = add_mat(Material::diffuse(Vec3(0.08f, 0.095f, 0.12f)));
        int m_silh       = add_mat(Material::diffuse(Vec3(0.03f, 0.035f, 0.045f)));
        int m_umb_glow   = add_mat(Material::emissive(Vec3(1.8f, 7.5f, 10.0f)));
        int m_umb_top    = add_mat(Material::metal(Vec3(0.10f, 0.18f, 0.26f), 0.18f));
        int m_rain_glint = add_mat(Material::metal(Vec3(0.75f, 0.88f, 0.98f), 0.05f));

        int m_bldg_cyn   = add_mat(Material::building_facade(Vec3(0.045f, 0.055f, 0.075f), Vec3(0.28f, 0.86f, 1.0f), 2.8f, 0.54f));
        int m_bldg_amb   = add_mat(Material::building_facade(Vec3(0.050f, 0.050f, 0.068f), Vec3(0.95f, 0.65f, 0.30f), 2.5f, 0.48f));
        int m_bldg_mag   = add_mat(Material::building_facade(Vec3(0.042f, 0.048f, 0.072f), Vec3(0.88f, 0.30f, 0.85f), 2.6f, 0.46f));

        int m_sign_cyn   = add_mat(Material::neon_sign(Vec3(0.15f, 0.92f, 1.0f), 6.2f, 2.3f));
        int m_sign_mag   = add_mat(Material::neon_sign(Vec3(1.0f, 0.18f, 0.72f), 6.4f, 2.3f));
        int m_sign_grn   = add_mat(Material::neon_sign(Vec3(0.18f, 1.0f, 0.58f), 5.8f, 2.5f));
        int m_sign_org   = add_mat(Material::neon_sign(Vec3(1.0f, 0.48f, 0.10f), 6.4f, 2.2f));
        int m_signs[4]   = {m_sign_cyn, m_sign_mag, m_sign_grn, m_sign_org};

        int m_bar_cyn    = add_mat(Material::emissive(Vec3(0.7f, 6.2f, 8.8f)));
        int m_bar_mag    = add_mat(Material::emissive(Vec3(8.5f, 0.9f, 5.4f)));
        int m_bar_wht    = add_mat(Material::emissive(Vec3(9.0f, 9.8f, 11.0f)));

        auto add_smooth_tri = [&](const Vec3& a, const Vec3& b, const Vec3& c,
                                  const Vec3& na, const Vec3& nb, const Vec3& nc, int mat_id) {
            if (cross(b - a, c - a).length_squared() < 1e-14f) return;
            out.prims.push_back(Primitive::make_smooth_triangle(a, b, c, na, nb, nc, mat_id));
            ++out.triangle_count;
        };
        auto add_quad = [&](const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, int mat_id) {
            add_tri(a, b, c, mat_id);
            add_tri(a, c, d, mat_id);
        };
        auto add_tube_seg = [&](const Vec3& p0, const Vec3& p1, float rad, int sides, int mat_id) {
            Vec3 axis = p1 - p0;
            if (axis.length_squared() < 1e-10f) return;
            Vec3 d = normalize(axis);
            Vec3 up = (fabsf(d.y) < 0.9f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
            Vec3 u = normalize(cross(up, d));
            Vec3 v = cross(d, u);
            for (int k = 0; k < sides; ++k) {
                float a0 = 2.0f * RT_PI * static_cast<float>(k) / static_cast<float>(sides);
                float a1 = 2.0f * RT_PI * static_cast<float>((k + 1) % sides) / static_cast<float>(sides);
                Vec3 n0 = cosf(a0) * u + sinf(a0) * v;
                Vec3 n1 = cosf(a1) * u + sinf(a1) * v;
                Vec3 v00 = p0 + rad * n0, v01 = p0 + rad * n1;
                Vec3 v10 = p1 + rad * n0, v11 = p1 + rad * n1;
                add_smooth_tri(v00, v10, v11, n0, n0, n1, mat_id);
                add_smooth_tri(v00, v11, v01, n0, n1, n1, mat_id);
            }
        };

        // 1. Rain-Slicked Asphalt Boulevard + Raised Concrete Sidewalks
        add_quad(Vec3(-20.0f, 0.0f, 20.0f), Vec3(20.0f, 0.0f, 20.0f),
                 Vec3(20.0f, 0.0f, -95.0f), Vec3(-20.0f, 0.0f, -95.0f), m_asphalt);
        add_box(Vec3(-16.0f, 0.01f, -92.0f), Vec3(-4.45f, 0.17f, 16.0f), m_sidewalk);
        add_box(Vec3( 4.45f, 0.01f, -92.0f), Vec3( 16.0f, 0.17f, 16.0f), m_sidewalk);
        // Subtle glowing cyan & magenta street-curb accents
        for (int s = 0; s < 14; ++s) {
            float sz = 10.0f - static_cast<float>(s) * 5.2f;
            add_box(Vec3(-4.46f, 0.15f, sz - 0.8f), Vec3(-4.42f, 0.19f, sz + 0.8f), (s & 1) ? m_bar_cyn : m_bar_mag);
            add_box(Vec3( 4.42f, 0.15f, sz - 0.8f), Vec3( 4.46f, 0.19f, sz + 0.8f), (s & 1) ? m_bar_mag : m_bar_cyn);
        }

        // 2. Hero 3D Widebody Cyberpunk Sports Car (Supra-style Coupe)
        const Vec3 car_pos(0.05f, 0.0f, 3.35f);
        const float car_yaw = 1.06f; // ~61 deg: nose points forward-right, widebody side & headlights face camera
        const float cy_c = cosf(car_yaw), sy_c = sinf(car_yaw);
        auto car_xform = [&](const Vec3& lp) -> Vec3 {
            return car_pos + Vec3(lp.x * cy_c + lp.z * sy_c, lp.y, -lp.x * sy_c + lp.z * cy_c);
        };
        auto car_vec = [&](const Vec3& lv) -> Vec3 {
            return Vec3(lv.x * cy_c + lv.z * sy_c, lv.y, -lv.x * sy_c + lv.z * cy_c);
        };
        auto car_box = [&](const Vec3& mn, const Vec3& mx, int mat_id) {
            Vec3 p[8] = {
                car_xform(Vec3(mn.x, mn.y, mn.z)), car_xform(Vec3(mx.x, mn.y, mn.z)),
                car_xform(Vec3(mx.x, mx.y, mn.z)), car_xform(Vec3(mn.x, mx.y, mn.z)),
                car_xform(Vec3(mn.x, mn.y, mx.z)), car_xform(Vec3(mx.x, mn.y, mx.z)),
                car_xform(Vec3(mx.x, mx.y, mx.z)), car_xform(Vec3(mn.x, mx.y, mx.z))
            };
            add_quad(p[4], p[5], p[6], p[7], mat_id); // +Z
            add_quad(p[1], p[0], p[3], p[2], mat_id); // -Z
            add_quad(p[5], p[1], p[2], p[6], mat_id); // +X
            add_quad(p[0], p[4], p[7], p[3], mat_id); // -X
            add_quad(p[7], p[6], p[2], p[3], mat_id); // +Y
            add_quad(p[0], p[1], p[5], p[4], mat_id); // -Y
        };

        // Helper to emit a smooth structured lofted grid with averaged vertex normals
        auto emit_smooth_grid = [&](const std::vector<std::vector<Vec3>>& grid, int mat_id, bool flip = false) {
            int nz = static_cast<int>(grid.size());
            int np = static_cast<int>(grid[0].size());
            std::vector<std::vector<Vec3>> norms(nz, std::vector<Vec3>(np, Vec3(0.0f)));
            for (int i = 0; i + 1 < nz; ++i) {
                for (int j = 0; j + 1 < np; ++j) {
                    Vec3 a = grid[i][j], b = grid[i + 1][j], d = grid[i][j + 1];
                    Vec3 fn = cross(b - a, d - a);
                    if (flip) fn = -fn;
                    norms[i][j] += fn; norms[i + 1][j] += fn;
                    norms[i + 1][j + 1] += fn; norms[i][j + 1] += fn;
                }
            }
            for (int i = 0; i < nz; ++i) {
                for (int j = 0; j < np; ++j) {
                    norms[i][j] = normalize(norms[i][j]);
                }
            }
            for (int i = 0; i + 1 < nz; ++i) {
                for (int j = 0; j + 1 < np; ++j) {
                    if (!flip) {
                        add_smooth_tri(grid[i][j], grid[i + 1][j], grid[i + 1][j + 1],
                                       norms[i][j], norms[i + 1][j], norms[i + 1][j + 1], mat_id);
                        add_smooth_tri(grid[i][j], grid[i + 1][j + 1], grid[i][j + 1],
                                       norms[i][j], norms[i + 1][j + 1], norms[i][j + 1], mat_id);
                    } else {
                        add_smooth_tri(grid[i][j], grid[i + 1][j + 1], grid[i + 1][j],
                                       norms[i][j], norms[i + 1][j + 1], norms[i + 1][j], mat_id);
                        add_smooth_tri(grid[i][j], grid[i][j + 1], grid[i + 1][j + 1],
                                       norms[i][j], norms[i][j + 1], norms[i + 1][j + 1], mat_id);
                    }
                }
            }
        };

        // 2a. Sculpted Lower Widebody Coupe Shell (24 stations x 11 profile points)
        const int NZ = 24;
        std::vector<std::vector<Vec3>> body_grid(NZ, std::vector<Vec3>(11));
        for (int i = 0; i < NZ; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(NZ - 1);
            float z = rt_lerp(-2.12f, 2.15f, t);

            // Muscular widebody fender flares around front (+1.34) & rear (-1.30) axles
            float f_flare = 0.13f * expf(-((z - 1.34f) * (z - 1.34f)) / 0.18f);
            float r_flare = 0.16f * expf(-((z + 1.30f) * (z + 1.30f)) / 0.22f);
            float nose_taper = (z > 1.55f) ? (1.0f - 0.16f * powf((z - 1.55f) / 0.62f, 1.8f)) : 1.0f;
            float tail_taper = (z < -1.65f) ? (1.0f - 0.08f * powf((-1.65f - z) / 0.50f, 1.6f)) : 1.0f;
            float hw = (0.87f + f_flare + r_flare) * nose_taper * tail_taper;

            float y_bot = (z > 1.7f) ? 0.13f : ((z < -1.7f) ? 0.18f : 0.15f);
            float y_belt;
            if (z > 0.72f) {
                float u = (z - 0.72f) / 1.43f;
                y_belt = rt_lerp(0.76f, 0.52f, u * u * 0.75f + 0.25f * u);
            } else if (z < -0.55f) {
                float u = (-0.55f - z) / 1.57f;
                y_belt = rt_lerp(0.76f, 0.85f, sinf(u * RT_PI * 0.5f));
            } else {
                y_belt = 0.75f;
            }
            float y_mid = rt_lerp(y_bot, y_belt, 0.45f);
            float deck_w = hw * 0.76f;
            float hood_crown = (z > 0.7f && z < 1.9f) ? 0.045f : 0.015f;

            // Cross-section from bottom-left -> left side -> deck -> right side -> bottom-right
            body_grid[i][0]  = car_xform(Vec3(-hw * 0.84f, y_bot, z));
            body_grid[i][1]  = car_xform(Vec3(-hw * 0.99f, y_bot + 0.10f, z));
            body_grid[i][2]  = car_xform(Vec3(-hw * 1.02f, y_mid, z));
            body_grid[i][3]  = car_xform(Vec3(-hw * 0.95f, y_belt - 0.06f, z));
            body_grid[i][4]  = car_xform(Vec3(-deck_w, y_belt, z));
            body_grid[i][5]  = car_xform(Vec3(0.0f, y_belt + hood_crown, z));
            body_grid[i][6]  = car_xform(Vec3( deck_w, y_belt, z));
            body_grid[i][7]  = car_xform(Vec3( hw * 0.95f, y_belt - 0.06f, z));
            body_grid[i][8]  = car_xform(Vec3( hw * 1.02f, y_mid, z));
            body_grid[i][9]  = car_xform(Vec3( hw * 0.99f, y_bot + 0.10f, z));
            body_grid[i][10] = car_xform(Vec3( hw * 0.84f, y_bot, z));
        }
        emit_smooth_grid(body_grid, m_car_body, false);

        // Front & rear lower bumper caps + dark carbon splitter & diffuser
        car_box(Vec3(-0.84f, 0.11f, 1.98f), Vec3(0.84f, 0.15f, 2.24f), m_car_carbon); // front splitter
        car_box(Vec3(-0.92f, 0.12f, -1.65f), Vec3(0.92f, 0.16f, 1.65f), m_car_carbon); // side skirts
        car_box(Vec3(-0.78f, 0.16f, -2.16f), Vec3(0.78f, 0.34f, -1.98f), m_car_carbon); // rear diffuser
        car_box(Vec3(-0.52f, 0.22f, 2.02f), Vec3(0.52f, 0.44f, 2.18f), m_car_carbon);   // front intercooler grille
        car_box(Vec3(-0.46f, 0.25f, 2.17f), Vec3(0.46f, 0.28f, 2.20f), m_amber_led);    // subtle amber intake glow

        // 2b. Fastback Greenhouse / Cabin (Windshield + Roof + Side/Rear Glass)
        const int NC = 13;
        std::vector<std::vector<Vec3>> glass_grid(NC, std::vector<Vec3>(7));
        std::vector<std::vector<Vec3>> roof_grid(6, std::vector<Vec3>(5));
        for (int i = 0; i < NC; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(NC - 1);
            float z = rt_lerp(-1.68f, 0.82f, t);
            float y_base = (z > 0.0f) ? rt_lerp(0.75f, 0.72f, z / 0.82f) : rt_lerp(0.75f, 0.83f, -z / 1.68f);
            // Smooth coupe arch peaking around z = -0.25
            float arch = sinf(rt_clamp((t + 0.06f) / 1.08f, 0.0f, 1.0f) * RT_PI);
            float y_top = y_base + 0.46f * powf(arch, 0.75f);
            float w_bot = rt_lerp(0.68f, 0.69f, arch);
            float w_top = rt_lerp(0.50f, 0.54f, arch);

            glass_grid[i][0] = car_xform(Vec3(-w_bot, y_base - 0.01f, z));
            glass_grid[i][1] = car_xform(Vec3(-rt_lerp(w_bot, w_top, 0.65f), rt_lerp(y_base, y_top, 0.65f), z));
            glass_grid[i][2] = car_xform(Vec3(-w_top, y_top, z));
            glass_grid[i][3] = car_xform(Vec3(0.0f, y_top + 0.035f, z));
            glass_grid[i][4] = car_xform(Vec3( w_top, y_top, z));
            glass_grid[i][5] = car_xform(Vec3( rt_lerp(w_bot, w_top, 0.65f), rt_lerp(y_base, y_top, 0.65f), z));
            glass_grid[i][6] = car_xform(Vec3( w_bot, y_base - 0.01f, z));
        }
        emit_smooth_grid(glass_grid, m_car_glass, false);

        // Carbon-paint roof cap + A/B pillars over the cabin
        for (int i = 0; i < 6; ++i) {
            float t = static_cast<float>(i) / 5.0f;
            float z = rt_lerp(-0.92f, 0.14f, t);
            float y_r = rt_lerp(1.13f, 1.15f, sinf(t * RT_PI));
            float wr = rt_lerp(0.49f, 0.52f, sinf(t * RT_PI));
            roof_grid[i][0] = car_xform(Vec3(-wr, y_r, z));
            roof_grid[i][1] = car_xform(Vec3(-wr * 0.5f, y_r + 0.04f, z));
            roof_grid[i][2] = car_xform(Vec3(0.0f, y_r + 0.055f, z));
            roof_grid[i][3] = car_xform(Vec3( wr * 0.5f, y_r + 0.04f, z));
            roof_grid[i][4] = car_xform(Vec3( wr, y_r, z));
        }
        emit_smooth_grid(roof_grid, m_car_carbon, false);

        // 2c. GT Rear Wing, Mirrors, Headlights, Taillights, Underbody Glow & Cyber Cables
        car_box(Vec3(-0.50f, 0.82f, -2.02f), Vec3(-0.44f, 1.16f, -1.86f), m_car_carbon);
        car_box(Vec3( 0.44f, 0.82f, -2.02f), Vec3( 0.50f, 1.16f, -1.86f), m_car_carbon);
        car_box(Vec3(-0.92f, 1.15f, -2.14f), Vec3( 0.92f, 1.20f, -1.82f), m_car_carbon); // wing blade
        car_box(Vec3(-0.94f, 1.09f, -2.16f), Vec3(-0.90f, 1.25f, -1.80f), m_car_carbon); // left endplate
        car_box(Vec3( 0.90f, 1.09f, -2.16f), Vec3( 0.94f, 1.25f, -1.80f), m_car_carbon); // right endplate

        // Side mirrors
        car_box(Vec3(-0.98f, 0.77f, 0.48f), Vec3(-0.80f, 0.86f, 0.64f), m_car_carbon);
        car_box(Vec3( 0.80f, 0.77f, 0.48f), Vec3( 0.98f, 0.86f, 0.64f), m_car_carbon);

        // Headlights (crisp dual LED bars) + Taillights + Dual Chrome Exhausts
        car_box(Vec3(-0.82f, 0.47f, 1.98f), Vec3(-0.48f, 0.56f, 2.13f), m_headlight);
        car_box(Vec3( 0.48f, 0.47f, 1.98f), Vec3( 0.82f, 0.56f, 2.13f), m_headlight);
        car_box(Vec3(-0.80f, 0.68f, -2.15f), Vec3(0.80f, 0.75f, -2.10f), m_taillight);
        add_tube_seg(car_xform(Vec3(-0.56f, 0.22f, -2.05f)), car_xform(Vec3(-0.56f, 0.22f, -2.20f)), 0.055f, 10, m_car_chrome);
        add_tube_seg(car_xform(Vec3( 0.56f, 0.22f, -2.05f)), car_xform(Vec3( 0.56f, 0.22f, -2.20f)), 0.055f, 10, m_car_chrome);

        // Subtle cyan underbody glow strip & cyberpunk exterior door cables
        car_box(Vec3(-0.74f, 0.09f, -1.40f), Vec3(0.74f, 0.12f, 1.40f), m_under_cyn);
        for (int seg = 0; seg < 6; ++seg) {
            float u0 = static_cast<float>(seg) / 6.0f;
            float u1 = static_cast<float>(seg + 1) / 6.0f;
            float z0 = rt_lerp(-0.95f, 0.65f, u0), z1 = rt_lerp(-0.95f, 0.65f, u1);
            float y0 = 0.56f - 0.18f * sinf(u0 * RT_PI), y1 = 0.56f - 0.18f * sinf(u1 * RT_PI);
            float x0 = -0.94f - 0.03f * sinf(u0 * RT_PI), x1 = -0.94f - 0.03f * sinf(u1 * RT_PI);
            add_tube_seg(car_xform(Vec3(x0, y0, z0)), car_xform(Vec3(x1, y1, z1)), 0.022f, 8, m_car_carbon);
            add_tube_seg(car_xform(Vec3(x0, y0 + 0.06f, z0)), car_xform(Vec3(x1, y1 + 0.06f, z1)), 0.016f, 8, m_car_carbon);
        }

        // 2d. Four Multi-Spoke Widebody Wheels (front wheels turned +0.18 rad)
        auto add_wheel = [&](const Vec3& lc, float steer_rad) {
            float cs = cosf(steer_rad), ss = sinf(steer_rad);
            auto wpt = [&](const Vec3& wp) -> Vec3 {
                Vec3 r(wp.x * cs + wp.z * ss, wp.y, -wp.x * ss + wp.z * cs);
                return car_xform(lc + r);
            };
            const int NSEG = 24;
            const float R_out = 0.34f;
            const float R_rim = 0.24f;
            const float hw_t = 0.145f;
            for (int k = 0; k < NSEG; ++k) {
                float a0 = 2.0f * RT_PI * static_cast<float>(k) / static_cast<float>(NSEG);
                float a1 = 2.0f * RT_PI * static_cast<float>((k + 1) % NSEG) / static_cast<float>(NSEG);
                float c0 = cosf(a0), s0 = sinf(a0);
                float c1 = cosf(a1), s1 = sinf(a1);
                Vec3 n0 = car_vec(Vec3(0.0f, s0, c0));
                Vec3 n1 = car_vec(Vec3(0.0f, s1, c1));

                // Tire tread
                Vec3 t00 = wpt(Vec3(-hw_t, R_out * s0, R_out * c0));
                Vec3 t01 = wpt(Vec3( hw_t, R_out * s0, R_out * c0));
                Vec3 t10 = wpt(Vec3(-hw_t, R_out * s1, R_out * c1));
                Vec3 t11 = wpt(Vec3( hw_t, R_out * s1, R_out * c1));
                add_smooth_tri(t00, t01, t11, n0, n0, n1, m_tire);
                add_smooth_tri(t00, t11, t10, n0, n1, n1, m_tire);

                // Outer & inner tire sidewalls
                Vec3 r0o = wpt(Vec3(-hw_t * 0.92f, R_rim * s0, R_rim * c0));
                Vec3 r1o = wpt(Vec3(-hw_t * 0.92f, R_rim * s1, R_rim * c1));
                add_quad(t00, t10, r1o, r0o, m_tire);
                Vec3 r0i = wpt(Vec3( hw_t * 0.92f, R_rim * s0, R_rim * c0));
                Vec3 r1i = wpt(Vec3( hw_t * 0.92f, R_rim * s1, R_rim * c1));
                add_quad(t11, t01, r0i, r1i, m_tire);

                // Deep-dish chrome inner lip
                Vec3 d0 = wpt(Vec3(-hw_t * 0.45f, (R_rim - 0.02f) * s0, (R_rim - 0.02f) * c0));
                Vec3 d1 = wpt(Vec3(-hw_t * 0.45f, (R_rim - 0.02f) * s1, (R_rim - 0.02f) * c1));
                add_quad(r0o, r1o, d1, d0, m_car_chrome);

                // Dark inner brake disc backing
                Vec3 hub = wpt(Vec3(-hw_t * 0.35f, 0.0f, 0.0f));
                add_tri(hub, d0, d1, m_car_carbon);
            }
            // 10 Chrome Spokes + Center Hub + Cyan Caliper
            Vec3 hub_l = wpt(Vec3(-hw_t * 0.72f, 0.0f, 0.0f));
            for (int sp = 0; sp < 10; ++sp) {
                float a = 2.0f * RT_PI * static_cast<float>(sp) / 10.0f;
                Vec3 rim_p = wpt(Vec3(-hw_t * 0.82f, (R_rim - 0.01f) * sinf(a), (R_rim - 0.01f) * cosf(a)));
                add_tube_seg(hub_l, rim_p, 0.014f, 6, m_car_chrome);
            }
            add_tube_seg(wpt(Vec3(-hw_t * 0.88f, 0.0f, 0.0f)), wpt(Vec3(-hw_t * 0.35f, 0.0f, 0.0f)), 0.055f, 10, m_car_chrome);
            add_tube_seg(wpt(Vec3(-hw_t * 0.55f, 0.11f, -0.07f)), wpt(Vec3(-hw_t * 0.55f, 0.15f, 0.07f)), 0.032f, 6, m_under_cyn);
        };
        add_wheel(Vec3(-0.91f, 0.34f,  1.34f),  0.18f);
        add_wheel(Vec3( 0.91f, 0.34f,  1.34f),  0.18f);
        add_wheel(Vec3(-0.94f, 0.34f, -1.30f),  0.0f);
        add_wheel(Vec3( 0.94f, 0.34f, -1.30f),  0.0f);

        // 3. Elevated Monorail Viaduct + Cyberpunk Train in Midground
        const float mono_y = 5.15f;
        const float mono_z0 = -16.5f;
        const float mono_z1 = -26.0f;
        // Twin sloped girder beam across the street
        add_quad(Vec3(-9.5f, mono_y, mono_z0), Vec3(9.5f, mono_y + 1.25f, mono_z1),
                 Vec3(9.5f, mono_y + 1.85f, mono_z1), Vec3(-9.5f, mono_y + 0.60f, mono_z0), m_concrete);
        add_quad(Vec3(-9.5f, mono_y, mono_z0 - 0.7f), Vec3(9.5f, mono_y + 1.25f, mono_z1 - 0.7f),
                 Vec3(9.5f, mono_y + 1.85f, mono_z1 - 0.7f), Vec3(-9.5f, mono_y + 0.60f, mono_z0 - 0.7f), m_concrete);
        // Support pillars on sidewalks
        add_box(Vec3(-5.1f, 0.17f, -18.8f), Vec3(-4.55f, mono_y + 0.45f, -18.2f), m_concrete);
        add_box(Vec3( 4.55f, 0.17f, -24.2f), Vec3( 5.1f, mono_y + 1.45f, -23.6f), m_concrete);
        // Sleek 4-car monorail train with illuminated cyan window strip
        for (int car = 0; car < 4; ++car) {
            float u0 = 0.08f + static_cast<float>(car) * 0.22f;
            float u1 = u0 + 0.19f;
            float x0 = rt_lerp(-8.0f, 8.0f, u0), x1 = rt_lerp(-8.0f, 8.0f, u1);
            float y0 = mono_y + 0.62f + u0 * 1.25f, y1 = mono_y + 0.62f + u1 * 1.25f;
            float z0 = rt_lerp(mono_z0, mono_z1, u0) - 0.1f, z1 = rt_lerp(mono_z0, mono_z1, u1) - 0.1f;
            add_quad(Vec3(x0, y0, z0), Vec3(x1, y1, z1), Vec3(x1, y1 + 0.85f, z1), Vec3(x0, y0 + 0.85f, z0), m_car_body);
            add_quad(Vec3(x0, y0 + 0.34f, z0 + 0.03f), Vec3(x1, y1 + 0.34f, z1 + 0.03f),
                     Vec3(x1, y1 + 0.62f, z1 + 0.03f), Vec3(x0, y0 + 0.62f, z0 + 0.03f), m_bar_cyn);
        }

        // 4. Skyscrapers, Storefronts, Neon Signs & Hero Billboards
        int bldg_mats[3] = {m_bldg_cyn, m_bldg_amb, m_bldg_mag};
        for (int row = 0; row < 12; ++row) {
            float zc = 8.5f - static_cast<float>(row) * 5.8f;
            for (int side = -1; side <= 1; side += 2) {
                float bw = rng.next_float(3.6f, 5.2f);
                float bd = rng.next_float(4.2f, 5.4f);
                float bh = rng.next_float(13.0f, 26.0f);
                float inner_edge = static_cast<float>(side) * (5.2f + rng.next_float(0.0f, 0.8f));
                float bx0 = (side < 0) ? (inner_edge - bw) : inner_edge;
                float bx1 = (side < 0) ? inner_edge : (inner_edge + bw);
                int bmat = bldg_mats[(row + (side > 0 ? 1 : 0)) % 3];
                add_box(Vec3(bx0, 0.17f, zc - 0.5f * bd), Vec3(bx1, bh, zc + 0.5f * bd), bmat);

                // Stepped upper tower + rooftop HVAC
                add_box(Vec3(bx0 + 0.5f, bh + 0.02f, zc - 0.35f * bd),
                        Vec3(bx1 - 0.5f, bh + rng.next_float(2.0f, 5.5f), zc + 0.35f * bd), m_steel);

                // Protruding vertical neon cyber-glyph sign on steel bracket
                float sx0 = (side < 0) ? (inner_edge + 0.25f) : (inner_edge - 0.55f);
                float sx1 = sx0 + 0.30f;
                float sh0 = rng.next_float(2.6f, 4.4f);
                float sh1 = sh0 + rng.next_float(3.2f, 6.0f);
                float sz = zc + rng.next_float(-1.2f, 1.2f);
                add_box(Vec3(sx0, sh0, sz - 0.06f), Vec3(sx1, sh1, sz + 0.06f), m_signs[(row * 3 + (side > 0 ? 1 : 0)) & 3]);
                // Steel mounting arm
                add_box(Vec3((side < 0) ? inner_edge : sx1, sh0 + 0.6f, sz - 0.04f),
                        Vec3((side < 0) ? sx0 : inner_edge, sh0 + 0.72f, sz + 0.04f), m_steel);

                // Horizontal storefront neon fascia above sidewalk
                if (row < 7) {
                    float fx0 = (side < 0) ? (inner_edge + 0.04f) : (inner_edge - 0.18f);
                    add_box(Vec3(fx0, 2.25f, zc - 1.3f), Vec3(fx0 + 0.14f, 2.95f, zc + 1.3f),
                            m_signs[(row + (side > 0 ? 2 : 0)) & 3]);
                    // Bright doorway/storefront interior glow
                    add_box(Vec3(fx0, 0.25f, zc - 0.9f), Vec3(fx0 + 0.06f, 2.0f, zc + 0.9f),
                            (row & 1) ? m_bar_cyn : m_bar_mag);
                }
            }
        }

        // Tall Background Skyline Towers rising behind the monorail into the misty sky
        for (int t = 0; t < 11; ++t) {
            float tx = -18.0f + static_cast<float>(t) * 3.6f + rng.next_float(-0.6f, 0.6f);
            float tz = -58.0f - rng.next_float(0.0f, 28.0f);
            float tw = rng.next_float(3.2f, 5.5f);
            float th = rng.next_float(24.0f, 44.0f);
            add_box(Vec3(tx - 0.5f * tw, 0.0f, tz - 2.5f), Vec3(tx + 0.5f * tw, th, tz + 2.5f), bldg_mats[t % 3]);
            // Rooftop neon crown
            add_box(Vec3(tx - 0.35f * tw, th - 0.9f, tz + 2.52f),
                    Vec3(tx + 0.35f * tw, th - 0.2f, tz + 2.68f), m_signs[t & 3]);
        }

        // Midground Illuminated Hero Billboard Screens (left & right towers)
        add_box(Vec3(-6.2f, 4.4f, -9.8f), Vec3(-5.95f, 8.6f, -6.2f), m_sign_cyn);
        add_box(Vec3( 5.95f, 5.2f, -12.6f), Vec3( 6.2f, 9.8f, -8.8f), m_sign_mag);
        // Bright streetlamp fixture on left sidewalk casting rim glow
        add_box(Vec3(-4.7f, 0.17f, -4.5f), Vec3(-4.56f, 5.2f, -4.36f), m_steel);
        add_box(Vec3(-4.6f, 5.05f, -4.48f), Vec3(-3.3f, 5.20f, -4.34f), m_steel);
        add_box(Vec3(-3.8f, 4.98f, -4.46f), Vec3(-3.3f, 5.05f, -4.36f), m_bar_wht);

        // 5. Five Sidewalk Pedestrians with Glowing Neon Umbrellas
        auto add_pedestrian = [&](const Vec3& base, float height, int umb_mat) {
            // Legs, torso, head
            add_tube_seg(base + Vec3(-0.09f, 0.0f, 0.0f), base + Vec3(-0.07f, height * 0.48f, 0.0f), 0.065f, 6, m_silh);
            add_tube_seg(base + Vec3( 0.09f, 0.0f, 0.0f), base + Vec3( 0.07f, height * 0.48f, 0.0f), 0.065f, 6, m_silh);
            add_tube_seg(base + Vec3(0.0f, height * 0.46f, 0.0f), base + Vec3(0.0f, height * 0.84f, 0.0f), 0.15f, 8, m_silh);
            add_sphere(base + Vec3(0.0f, height * 0.93f, 0.0f), 0.105f, m_silh);
            // Umbrella handle & glowing dome canopy
            Vec3 u_top = base + Vec3(0.08f, height + 0.34f, 0.0f);
            add_tube_seg(base + Vec3(0.12f, height * 0.65f, 0.0f), u_top, 0.015f, 5, m_steel);
            const int USEG = 12;
            const float urad = 0.56f;
            const float udrop = 0.18f;
            for (int k = 0; k < USEG; ++k) {
                float a0 = 2.0f * RT_PI * static_cast<float>(k) / static_cast<float>(USEG);
                float a1 = 2.0f * RT_PI * static_cast<float>((k + 1) % USEG) / static_cast<float>(USEG);
                Vec3 r0(urad * cosf(a0), -udrop, urad * sinf(a0));
                Vec3 r1(urad * cosf(a1), -udrop, urad * sinf(a1));
                Vec3 n_up(0.0f, 1.0f, 0.0f);
                Vec3 n0 = normalize(Vec3(cosf(a0), 0.5f, sinf(a0)));
                Vec3 n1 = normalize(Vec3(cosf(a1), 0.5f, sinf(a1)));
                add_smooth_tri(u_top, u_top + r1, u_top + r0, n_up, n1, n0, umb_mat);
                add_smooth_tri(u_top + Vec3(0.0f, 0.012f, 0.0f), u_top + r0 + Vec3(0.0f, 0.012f, 0.0f),
                               u_top + r1 + Vec3(0.0f, 0.012f, 0.0f), n_up, n0, n1, m_umb_top);
            }
        };
        add_pedestrian(Vec3(-4.85f, 0.17f, -0.5f), 1.66f, m_umb_glow);
        add_pedestrian(Vec3(-5.10f, 0.17f, -6.4f), 1.62f, m_umb_glow);
        add_pedestrian(Vec3(-4.80f, 0.17f, -12.5f), 1.64f, m_umb_glow);
        add_pedestrian(Vec3( 4.85f, 0.17f,  0.2f), 1.68f, m_umb_glow);
        add_pedestrian(Vec3( 5.05f, 0.17f, -7.2f), 1.60f, m_umb_glow);

        // 6. Overhead Power Cables & Fine Rain Streaks (scaled cleanly by --objects)
        for (int c = 0; c < 12; ++c) {
            float cz0 = 5.0f - static_cast<float>(c) * 3.4f;
            float cz1 = cz0 - rng.next_float(1.2f, 4.5f);
            float cy0 = rng.next_float(6.2f, 9.6f);
            float cy1 = rng.next_float(6.2f, 9.6f);
            add_tube_seg(Vec3(-5.2f, cy0, cz0), Vec3(5.2f, cy1, cz1), 0.016f, 4, m_silh);
        }

        int extra_detail = objects > 0 ? objects : 1500;
        // Street bollards & AC units
        int ac_count = rt_min(160, extra_detail / 8);
        for (int a = 0; a < ac_count; ++a) {
            float side = (a & 1) ? 1.0f : -1.0f;
            float az = rng.next_float(-40.0f, 8.0f);
            float ay = rng.next_float(2.4f, 11.0f);
            float ax = side * 5.15f;
            add_box(Vec3(ax - 0.25f, ay, az - 0.22f), Vec3(ax + 0.25f, ay + 0.35f, az + 0.22f), m_steel);
        }
        // Subtle glinting rain streaks (no spheres!)
        int rain_count = rt_max(0, extra_detail - ac_count);
        for (int r = 0; r < rain_count; ++r) {
            float rx = rng.next_float(-7.5f, 7.5f);
            float rz = rng.next_float(-28.0f, 7.2f);
            float ry = rng.next_float(0.25f, 6.8f);
            // Keep immediate lens area clear
            if ((Vec3(rx, ry, rz) - lookfrom).length() < 1.8f) continue;
            Vec3 p0(rx, ry, rz);
            Vec3 p1(rx + 0.014f, ry + 0.14f, rz);
            Vec3 p2(rx + 0.022f, ry + 0.14f, rz + 0.004f);
            add_tri(p0, p1, p2, m_rain_glint);
        }
    } else if (preset == "cyber_cathedral") {
        // Holographic Prism Vault & Nested Glass Megastructure
        out.sky_top = Vec3(0.008f, 0.014f, 0.045f);
        out.sky_bottom = Vec3(0.032f, 0.068f, 0.14f);
        out.sun_dir = normalize(Vec3(0.0f, 0.35f, -1.0f));
        out.sun_color = Vec3(0.4f, 0.85f, 1.0f);
        out.sun_intensity = 0.9f;

        Vec3 lookfrom(0.0f, 2.55f, 10.8f);
        Vec3 lookat(0.0f, 1.85f, -2.0f);
        out.cam = Camera::create(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 39.0f, aspect, 0.0f, 12.0f);

        int m_floor     = add_mat(Material::checker(Vec3(0.025f, 0.032f, 0.052f), Vec3(0.075f, 0.095f, 0.145f), 1.0f, 0.52f));
        int m_titanium  = add_mat(Material::metal(Vec3(0.24f, 0.28f, 0.36f), 0.06f));
        int m_chrome    = add_mat(Material::metal(Vec3(0.92f, 0.95f, 1.0f), 0.015f));
        int m_gold      = add_mat(Material::metal(Vec3(0.96f, 0.72f, 0.34f), 0.04f));
        int m_glass_o   = add_mat(Material::glass(1.52f, Vec3(0.84f, 0.98f, 1.0f)));
        int m_glass_i   = add_mat(Material::glass(1.66f, Vec3(1.0f, 0.82f, 0.98f)));
        int m_core_cyn  = add_mat(Material::emissive(Vec3(0.7f, 4.2f, 5.5f)));
        int m_cyn_emit  = add_mat(Material::emissive(Vec3(0.8f, 7.2f, 9.2f)));
        int m_mag_emit  = add_mat(Material::emissive(Vec3(8.5f, 1.0f, 5.2f)));
        int m_amb_emit  = add_mat(Material::emissive(Vec3(8.0f, 4.5f, 0.8f)));

        add_sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, m_floor);

        for (int p = 0; p < 7; ++p) {
            float pz = 6.0f - static_cast<float>(p) * 3.8f;
            add_box(Vec3(-4.6f, 0.01f, pz - 0.35f), Vec3(-3.92f, 6.78f, pz + 0.35f), m_titanium);
            add_box(Vec3( 3.92f, 0.01f, pz - 0.35f), Vec3( 4.6f, 6.78f, pz + 0.35f), m_titanium);
            add_box(Vec3(-3.90f, 0.6f, pz - 0.12f), Vec3(-3.76f, 6.2f, pz + 0.12f), (p & 1) ? m_mag_emit : m_cyn_emit);
            add_box(Vec3( 3.76f, 0.6f, pz - 0.12f), Vec3( 3.90f, 6.2f, pz + 0.12f), (p & 1) ? m_cyn_emit : m_mag_emit);
            add_box(Vec3(-4.2f, 6.80f, pz - 0.25f), Vec3(4.2f, 7.20f, pz + 0.25f), m_titanium);
            add_box(Vec3(-3.5f, 6.62f, pz - 0.10f), Vec3(3.5f, 6.78f, pz + 0.10f), (p & 1) ? m_cyn_emit : m_mag_emit);
        }

        // Nested refractive glass centerpiece + glowing core
        add_sphere(Vec3(0.0f, 2.05f, 0.0f), 2.00f, m_glass_o);
        add_sphere(Vec3(0.0f, 2.05f, 0.0f), 1.35f, m_glass_i);
        add_sphere(Vec3(0.0f, 2.05f, 0.0f), 0.44f, m_core_cyn);
        add_sphere(Vec3(-2.55f, 1.05f, 2.2f), 1.05f, m_chrome);
        add_sphere(Vec3( 2.55f, 1.05f, 2.2f), 1.05f, m_gold);

        int count = objects > 0 ? objects : 1200;
        for (int i = 0; i < count; ++i) {
            float fi = static_cast<float>(i);
            float ang = fi * 2.3999632f;
            float ring = 2.4f + 0.24f * sqrtf(fi + 1.0f);
            float x = cosf(ang) * rt_min(ring, 3.60f);
            float z = sinf(ang) * ring - 2.0f;
            if (z > 8.2f || z < -20.0f) z = rng.next_float(-18.0f, 7.5f);
            float r = rng.next_float(0.11f, 0.26f);
            float y = (i % 3 == 0) ? rng.next_float(1.6f, 5.4f) : (r + 0.01f);

            if ((Vec3(x, y, z) - Vec3(0.0f, 2.05f, 0.0f)).length() < 2.25f) continue;

            int mod = i % 5;
            if (mod == 0 || mod == 1) {
                add_crystal_shard(Vec3(x, y + r * 0.5f, z), r * 0.65f, r * 1.6f, r * 0.65f, ang, (mod == 0) ? m_glass_o : m_glass_i);
            } else if (mod == 2) {
                add_sphere(Vec3(x, y, z), r, m_chrome);
            } else if (mod == 3) {
                add_sphere(Vec3(x, y, z), r, m_glass_o);
            } else {
                int em = (i % 3 == 0) ? m_amb_emit : ((i & 1) ? m_cyn_emit : m_mag_emit);
                add_sphere(Vec3(x, y, z), r * 0.58f, em);
            }
        }
    } else if (preset == "orbital_megastructure") {
        // Deep-Space Dyson Ring, Glass Shell & Crystal Asteroid Field
        out.sky_top = Vec3(0.001f, 0.002f, 0.007f);
        out.sky_bottom = Vec3(0.010f, 0.025f, 0.055f);
        out.sun_dir = normalize(Vec3(0.55f, 0.45f, -0.7f));
        out.sun_color = Vec3(0.35f, 0.78f, 1.0f);
        out.sun_intensity = 1.4f;

        Vec3 lookfrom(0.0f, 4.4f, 12.8f);
        Vec3 lookat(0.0f, 0.55f, 0.0f);
        out.cam = Camera::create(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 38.0f, aspect, 0.0f, 13.0f);

        int m_hull      = add_mat(Material::metal(Vec3(0.28f, 0.32f, 0.40f), 0.08f));
        int m_chrome    = add_mat(Material::metal(Vec3(0.90f, 0.94f, 0.98f), 0.02f));
        int m_gold      = add_mat(Material::metal(Vec3(0.95f, 0.74f, 0.32f), 0.05f));
        int m_glass_p   = add_mat(Material::glass(1.52f, Vec3(0.84f, 0.96f, 1.0f)));
        int m_glass_m   = add_mat(Material::glass(1.58f, Vec3(1.0f, 0.80f, 0.95f)));
        int m_star_core = add_mat(Material::emissive(Vec3(9.5f, 4.8f, 1.4f)));
        int m_cyn_glow  = add_mat(Material::emissive(Vec3(0.8f, 6.8f, 9.2f)));
        int m_mag_glow  = add_mat(Material::emissive(Vec3(8.5f, 0.9f, 5.4f)));

        add_sphere(Vec3(0.0f, 0.6f, 0.0f), 2.15f, m_glass_p);
        add_sphere(Vec3(0.0f, 0.6f, 0.0f), 0.95f, m_star_core);

        const int ring_segs = 32;
        for (int s = 0; s < ring_segs; ++s) {
            float ang = (2.0f * RT_PI * static_cast<float>(s)) / static_cast<float>(ring_segs);
            float rad = 4.5f;
            float rx = cosf(ang) * rad;
            float rz = sinf(ang) * rad;
            float ry = 0.6f + 1.25f * sinf(ang + 0.35f);
            add_box(Vec3(rx - 0.34f, ry - 0.20f, rz - 0.34f),
                    Vec3(rx + 0.34f, ry + 0.20f, rz + 0.34f),
                    (s & 1) ? m_hull : m_chrome);
            add_box(Vec3(rx - 0.14f, ry + 0.22f, rz - 0.14f),
                    Vec3(rx + 0.14f, ry + 0.36f, rz + 0.14f),
                    (s % 3 == 0) ? m_mag_glow : m_cyn_glow);
        }

        int count = objects > 0 ? objects : 2000;
        for (int i = 0; i < count; ++i) {
            float fi = static_cast<float>(i);
            float ang = fi * 2.3999632f + rng.next_float(-0.1f, 0.1f);
            float rad = 3.2f + 0.26f * sqrtf(fi + 1.0f);
            float x = cosf(ang) * rad;
            float z = sinf(ang) * rad;
            float y = 0.6f + rng.next_float(-2.8f, 2.8f) + 0.85f * sinf(ang * 2.0f);

            // Keep central planet/ring corridor clear toward camera
            if (z > 2.0f && fabsf(x) < 3.6f && fabsf(y - 1.2f) < 2.2f) continue;
            if ((Vec3(x, y, z) - Vec3(0.0f, 0.6f, 0.0f)).length() < 2.65f) continue;
            if ((Vec3(x, y, z) - lookfrom).length() < 3.2f) continue;

            float r = rng.next_float(0.07f, 0.24f);
            int pick = i % 6;
            if (pick <= 1) {
                add_crystal_shard(Vec3(x, y, z), r * 0.8f, r * 1.6f, r * 0.8f, ang, (pick == 0) ? m_glass_p : m_glass_m);
            } else if (pick == 2) {
                add_crystal_shard(Vec3(x, y, z), r, r * 1.4f, r, ang * 1.7f, m_hull);
            } else if (pick == 3) {
                add_sphere(Vec3(x, y, z), r, m_chrome);
            } else if (pick == 4) {
                add_sphere(Vec3(x, y, z), r, m_gold);
            } else {
                add_sphere(Vec3(x, y, z), r * 0.55f, (i & 1) ? m_cyn_glow : m_mag_glow);
            }
        }
    } else {
        // Default: large_world (Ray Tracing in One Weekend inspired large procedural scene)
        out.sky_top = Vec3(0.32f, 0.56f, 0.92f);
        out.sky_bottom = Vec3(0.86f, 0.92f, 0.99f);
        out.sun_dir = normalize(Vec3(0.6f, 0.65f, 0.45f));
        out.sun_color = Vec3(1.0f, 0.93f, 0.78f);
        out.sun_intensity = 4.5f;

        Vec3 lookfrom(11.5f, 2.3f, 4.2f);
        Vec3 lookat(0.0f, 0.85f, 0.0f);
        out.cam = Camera::create(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 26.0f, aspect, 0.03f, 10.5f);

        int m_ground = add_mat(Material::checker(Vec3(0.18f, 0.20f, 0.23f), Vec3(0.84f, 0.86f, 0.88f), 1.15f, 0.10f));
        int m_glass  = add_mat(Material::glass(1.5f));
        int m_terra  = add_mat(Material::diffuse(Vec3(0.68f, 0.24f, 0.16f)));
        int m_metal  = add_mat(Material::metal(Vec3(0.86f, 0.76f, 0.56f), 0.02f));
        int m_glow   = add_mat(Material::emissive(Vec3(9.0f, 7.2f, 4.2f)));

        add_sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, m_ground);

        // 3 hero spheres
        add_sphere(Vec3( 0.0f, 1.0f, 0.0f), 1.0f, m_glass);
        add_sphere(Vec3(-3.2f, 1.0f, 0.0f), 1.0f, m_terra);
        add_sphere(Vec3( 3.2f, 1.0f, 0.0f), 1.0f, m_metal);

        int count = objects > 0 ? objects : 500;
        for (int i = 0; i < count; ++i) {
            // Golden-angle spiral distribution scales cleanly from 100 to 10,000+ objects
            float fi = static_cast<float>(i);
            float angle = fi * 2.3999632f + rng.next_float(-0.08f, 0.08f);
            float spread = 1.4f + 0.55f * sqrtf(fi + 1.0f);
            float r = rng.next_float(0.14f, 0.30f);
            float x = cosf(angle) * spread;
            float z = sinf(angle) * spread;

            // Keep hero spheres clear
            Vec3 c(x, r, z);
            if ((c - Vec3(0.0f, 1.0f, 0.0f)).length() < 1.35f) continue;
            if ((c - Vec3(-3.2f, 1.0f, 0.0f)).length() < 1.35f) continue;
            if ((c - Vec3(3.2f, 1.0f, 0.0f)).length() < 1.35f) continue;

            float choose = rng.next_float();
            int mat_id;
            if (choose < 0.62f) {
                Vec3 albedo(
                    rng.next_float(0.1f, 0.9f) * rng.next_float(0.2f, 0.95f),
                    rng.next_float(0.1f, 0.9f) * rng.next_float(0.2f, 0.95f),
                    rng.next_float(0.1f, 0.9f) * rng.next_float(0.2f, 0.95f)
                );
                mat_id = add_mat(Material::diffuse(albedo));
            } else if (choose < 0.84f) {
                Vec3 albedo(
                    rng.next_float(0.55f, 0.98f),
                    rng.next_float(0.55f, 0.98f),
                    rng.next_float(0.55f, 0.98f)
                );
                float fuzz = rng.next_float(0.0f, 0.28f);
                mat_id = add_mat(Material::metal(albedo, fuzz));
            } else if (choose < 0.96f) {
                mat_id = m_glass;
            } else {
                mat_id = m_glow;
            }

            add_sphere(c, r, mat_id);
        }
    }
}
