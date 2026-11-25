#pragma once

#include "scene.h"

RT_HD RT_INLINE Vec3 eval_sky(const SceneView& scene, const Vec3& dir) {
    Vec3 unit_dir = normalize(dir);
    float t = 0.5f * (unit_dir.y + 1.0f);
    Vec3 bg = (1.0f - t) * scene.sky_bottom + t * scene.sky_top;
    if (scene.sun_intensity > 0.0f) {
        float sun_cos = rt_max(0.0f, dot(unit_dir, scene.sun_dir));
        float glow = powf(sun_cos, 16.0f) * 0.25f + powf(sun_cos, 192.0f);
        bg += scene.sun_color * (glow * scene.sun_intensity);
    }
    return bg;
}

RT_HD RT_INLINE Vec3 trace_ray(
    const SceneView& scene,
    Ray r,
    int max_depth,
    Rng& rng,
    uint32_t& rays_traced
) {
    Vec3 accum(0.0f);
    Vec3 throughput(1.0f);

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        ++rays_traced;
        HitRecord rec;
        if (!bvh_intersect(scene.nodes, scene.node_count, scene.prims, r, RT_EPS, RT_INF, rec)) {
            accum += throughput * eval_sky(scene, r.dir);
            break;
        }

        // Optional atmospheric rainy night mist on primary/aerial segments
        if (scene.fog_density > 0.0f && rec.t < 200.0f) {
            float fog_f = 1.0f - expf(-scene.fog_density * rec.t);
            fog_f = rt_clamp(fog_f, 0.0f, 0.86f);
            accum += throughput * (fog_f * scene.fog_color);
            throughput *= (1.0f - fog_f);
        }

        const Material& mat = scene.mats[rec.mat_id];
        Vec3 attenuation(0.0f);
        Vec3 emitted(0.0f);
        Ray scattered;
        if (!scatter_material(mat, r, rec.p, rec.normal, rec.front_face, rng, attenuation, emitted, scattered)) {
            accum += throughput * emitted;
            break;
        }

        accum += throughput * emitted;
        throughput *= attenuation;

        float max_c = rt_max(throughput.x, rt_max(throughput.y, throughput.z));
        if (max_c < 1e-4f) break;

        r = scattered;
    }

    // Clamp extreme specular/glass fireflies
    accum.x = rt_min(accum.x, 8.0f);
    accum.y = rt_min(accum.y, 8.0f);
    accum.z = rt_min(accum.z, 8.0f);
    return accum;
}

RT_HD RT_INLINE Vec3 aces_tonemap(const Vec3& x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    Vec3 num(
        x.x * (a * x.x + b),
        x.y * (a * x.y + b),
        x.z * (a * x.z + b)
    );
    Vec3 den(
        x.x * (c * x.x + d) + e,
        x.y * (c * x.y + d) + e,
        x.z * (c * x.z + d) + e
    );
    return Vec3(
        rt_clamp(num.x / den.x, 0.0f, 1.0f),
        rt_clamp(num.y / den.y, 0.0f, 1.0f),
        rt_clamp(num.z / den.z, 0.0f, 1.0f)
    );
}

RT_HD RT_INLINE void render_pixel(
    const SceneView& scene,
    int x,
    int y,
    int width,
    int height,
    int spp,
    int max_depth,
    uint32_t seed,
    uint8_t* rgb_out,
    uint64_t& rays_out
) {
    Vec3 color(0.0f);
    uint32_t local_rays = 0;

    float inv_w = width > 1 ? (1.0f / static_cast<float>(width - 1)) : 1.0f;
    float inv_h = height > 1 ? (1.0f / static_cast<float>(height - 1)) : 1.0f;

    for (int s = 0; s < spp; ++s) {
        Rng rng = Rng::from_pixel(seed, x, y, s);
        float dx = (spp > 1) ? rng.next_float() : 0.5f;
        float dy = (spp > 1) ? rng.next_float() : 0.5f;

        float u = (static_cast<float>(x) + dx) * inv_w;
        float v = (static_cast<float>(height - 1 - y) + dy) * inv_h;

        Ray r = scene.cam.get_ray(u, v, rng);
        color += trace_ray(scene, r, max_depth, rng, local_rays);
    }

    rays_out += static_cast<uint64_t>(local_rays);

    float inv_spp = 1.0f / static_cast<float>(rt_max(1, spp));
    color *= inv_spp;
    color = aces_tonemap(color);

    const float inv_gamma = 1.0f / 2.2f;
    float r_g = powf(rt_clamp(color.x, 0.0f, 1.0f), inv_gamma);
    float g_g = powf(rt_clamp(color.y, 0.0f, 1.0f), inv_gamma);
    float b_g = powf(rt_clamp(color.z, 0.0f, 1.0f), inv_gamma);

    rgb_out[0] = static_cast<uint8_t>(rt_clamp(r_g * 255.0f + 0.5f, 0.0f, 255.0f));
    rgb_out[1] = static_cast<uint8_t>(rt_clamp(g_g * 255.0f + 0.5f, 0.0f, 255.0f));
    rgb_out[2] = static_cast<uint8_t>(rt_clamp(b_g * 255.0f + 0.5f, 0.0f, 255.0f));
}
