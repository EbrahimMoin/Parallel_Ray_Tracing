#pragma once

#include "ray.h"
#include "rng.h"

enum MatType {
    MAT_DIFFUSE         = 0,
    MAT_METAL           = 1,
    MAT_GLASS           = 2,
    MAT_EMISSIVE        = 3,
    MAT_CHECKER         = 4,
    MAT_WET_ASPHALT     = 5,
    MAT_CAR_PAINT       = 6,
    MAT_BUILDING_FACADE = 7,
    MAT_NEON_SIGN       = 8
};

RT_HD RT_INLINE float mat_hash2(int ix, int iz) {
    uint32_t h = static_cast<uint32_t>(ix) * 374761393u + static_cast<uint32_t>(iz) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return static_cast<float>((h ^ (h >> 16u)) & 0xFFFFFFu) * (1.0f / 16777216.0f);
}

RT_HD RT_INLINE float mat_vnoise2(float x, float z) {
    int ix = static_cast<int>(floorf(x));
    int iz = static_cast<int>(floorf(z));
    float fx = x - static_cast<float>(ix);
    float fz = z - static_cast<float>(iz);
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uz = fz * fz * (3.0f - 2.0f * fz);
    float a = mat_hash2(ix, iz);
    float b = mat_hash2(ix + 1, iz);
    float c = mat_hash2(ix, iz + 1);
    float d = mat_hash2(ix + 1, iz + 1);
    return rt_lerp(rt_lerp(a, b, ux), rt_lerp(c, d, ux), uz);
}

RT_HD RT_INLINE float mat_fbm2(float x, float z) {
    return 0.55f * mat_vnoise2(x, z)
         + 0.30f * mat_vnoise2(x * 2.13f + 5.1f, z * 2.13f - 3.7f)
         + 0.15f * mat_vnoise2(x * 4.31f - 2.4f, z * 4.31f + 7.9f);
}

struct Material {
    int   type;
    Vec3  albedo;
    Vec3  albedo2;
    Vec3  emission;
    float param;

    RT_HD Material()
        : type(MAT_DIFFUSE),
          albedo(Vec3(0.8f)),
          albedo2(Vec3(0.2f)),
          emission(Vec3(0.0f)),
          param(0.0f) {}

    RT_HD static Material diffuse(const Vec3& col) {
        Material m;
        m.type = MAT_DIFFUSE;
        m.albedo = col;
        return m;
    }

    RT_HD static Material metal(const Vec3& col, float fuzz) {
        Material m;
        m.type = MAT_METAL;
        m.albedo = col;
        m.param = rt_clamp(fuzz, 0.0f, 1.0f);
        return m;
    }

    RT_HD static Material glass(float ior, const Vec3& tint = Vec3(1.0f, 1.0f, 1.0f)) {
        Material m;
        m.type = MAT_GLASS;
        m.albedo = tint;
        m.param = ior;
        return m;
    }

    RT_HD static Material emissive(const Vec3& emit_col) {
        Material m;
        m.type = MAT_EMISSIVE;
        m.albedo = Vec3(0.0f);
        m.emission = emit_col;
        return m;
    }

    RT_HD static Material checker(const Vec3& c1, const Vec3& c2, float scale = 1.5f, float gloss = 0.12f) {
        Material m;
        m.type = MAT_CHECKER;
        m.albedo = c1;
        m.albedo2 = c2;
        m.emission = Vec3(scale, 0.0f, 0.0f);
        m.param = gloss;
        return m;
    }

    RT_HD static Material wet_asphalt(const Vec3& tarmac_col, float puddle_coverage = 0.55f) {
        Material m;
        m.type = MAT_WET_ASPHALT;
        m.albedo = tarmac_col;
        m.param = puddle_coverage;
        return m;
    }

    RT_HD static Material car_paint(const Vec3& base_col, float metallic_fuzz = 0.10f) {
        Material m;
        m.type = MAT_CAR_PAINT;
        m.albedo = base_col;
        m.param = metallic_fuzz;
        return m;
    }

    RT_HD static Material building_facade(
        const Vec3& hull_col,
        const Vec3& win_tint,
        float win_intensity = 3.2f,
        float win_density = 0.52f
    ) {
        Material m;
        m.type = MAT_BUILDING_FACADE;
        m.albedo = hull_col;
        m.albedo2 = win_tint;
        m.emission = Vec3(win_intensity, 0.0f, 0.0f);
        m.param = win_density;
        return m;
    }

    RT_HD static Material neon_sign(const Vec3& neon_col, float intensity = 6.5f, float glyph_scale = 2.4f) {
        Material m;
        m.type = MAT_NEON_SIGN;
        m.albedo = neon_col;
        m.emission = Vec3(intensity, 0.0f, 0.0f);
        m.param = glyph_scale;
        return m;
    }
};

RT_HD RT_INLINE bool scatter_material(
    const Material& mat,
    const Ray& r_in,
    const Vec3& p,
    const Vec3& normal,
    bool front_face,
    Rng& rng,
    Vec3& attenuation,
    Vec3& emitted,
    Ray& scattered
) {
    emitted = Vec3(0.0f);

    if (mat.type == MAT_EMISSIVE) {
        emitted = mat.emission;
        attenuation = Vec3(0.0f);
        return false;
    }

    if (mat.type == MAT_DIFFUSE) {
        Vec3 scatter_dir = normal + rng.random_unit_vector();
        if (scatter_dir.near_zero()) scatter_dir = normal;
        scattered = Ray(p + normal * RT_EPS, normalize(scatter_dir));
        attenuation = mat.albedo;
        return true;
    }

    if (mat.type == MAT_CHECKER) {
        float scale = mat.emission.x > 0.0f ? mat.emission.x : 1.5f;
        int ix = static_cast<int>(floorf(p.x * scale));
        int iz = static_cast<int>(floorf(p.z * scale));
        bool even = ((ix + iz) & 1) == 0;
        attenuation = even ? mat.albedo : mat.albedo2;

        if (mat.param > 0.0f && rng.next_float() < mat.param) {
            Vec3 refl = reflect(normalize(r_in.dir), normal);
            float gloss_fuzz = (mat.param > 0.35f) ? 0.035f : 0.06f;
            Vec3 fuzz_dir = refl + gloss_fuzz * rng.random_unit_vector();
            if (dot(fuzz_dir, normal) <= 0.0f) fuzz_dir = refl;
            scattered = Ray(p + normal * RT_EPS, normalize(fuzz_dir));
        } else {
            Vec3 scatter_dir = normal + rng.random_unit_vector();
            if (scatter_dir.near_zero()) scatter_dir = normal;
            scattered = Ray(p + normal * RT_EPS, normalize(scatter_dir));
        }
        return true;
    }

    if (mat.type == MAT_WET_ASPHALT) {
        float puddle_n = mat_fbm2(p.x * 0.45f + 1.3f, p.z * 0.32f - 2.1f);
        float thresh = 1.0f - rt_clamp(mat.param, 0.1f, 0.9f);
        float wet = rt_clamp((puddle_n - (thresh - 0.07f)) * 7.0f, 0.0f, 1.0f);

        // Micro-normal perturbation: subtle rain ripples in puddles, tarmac grit outside
        float nx = mat_vnoise2(p.x * 7.5f, p.z * 7.5f) - 0.5f;
        float nz = mat_vnoise2(p.x * 7.5f + 19.3f, p.z * 7.5f - 11.7f) - 0.5f;
        float ripple = sinf((p.x * p.x * 0.3f + p.z * p.z * 0.3f) * 2.5f + mat_vnoise2(p.x * 2.2f, p.z * 2.2f) * 8.0f) * 0.015f;
        float pert = rt_lerp(0.11f, 0.016f, wet);
        Vec3 n_pert = normalize(normal + Vec3(nx * pert + ripple * wet, 0.0f, nz * pert));

        Vec3 unit_dir = normalize(r_in.dir);
        float cos_theta = rt_clamp(dot(-unit_dir, n_pert), 0.0f, 1.0f);
        float f0 = rt_lerp(0.12f, 0.42f, wet);
        float fresnel = f0 + (1.0f - f0) * powf(1.0f - cos_theta, 3.5f);

        if (rng.next_float() < fresnel) {
            Vec3 refl = reflect(unit_dir, n_pert);
            float fuzz = rt_lerp(0.16f, 0.014f, wet);
            Vec3 dir = refl + fuzz * rng.random_unit_vector();
            if (dot(dir, normal) <= 0.0f) dir = reflect(unit_dir, normal);
            scattered = Ray(p + normal * RT_EPS, normalize(dir));
            attenuation = Vec3(rt_lerp(0.55f, 0.94f, wet));
        } else {
            Vec3 scatter_dir = n_pert + rng.random_unit_vector();
            if (dot(scatter_dir, normal) <= 0.0f) scatter_dir = normal + rng.random_unit_vector();
            scattered = Ray(p + normal * RT_EPS, normalize(scatter_dir));
            attenuation = mat.albedo * rt_lerp(1.0f, 0.35f, wet);
        }
        return true;
    }

    if (mat.type == MAT_CAR_PAINT) {
        Vec3 unit_dir = normalize(r_in.dir);
        float cos_theta = rt_clamp(dot(-unit_dir, normal), 0.0f, 1.0f);
        float fresnel = 0.24f + 0.76f * powf(1.0f - cos_theta, 4.0f);
        Vec3 refl = reflect(unit_dir, normal);
        if (rng.next_float() < fresnel) {
            Vec3 dir = refl + 0.022f * rng.random_unit_vector();
            if (dot(dir, normal) <= 0.0f) dir = refl;
            scattered = Ray(p + normal * RT_EPS, normalize(dir));
            attenuation = Vec3(0.94f, 0.96f, 0.99f);
        } else {
            Vec3 dir = refl + mat.param * rng.random_unit_vector();
            if (dot(dir, normal) <= 0.0f) {
                dir = normal + rng.random_unit_vector();
                if (dir.near_zero()) dir = normal;
            }
            scattered = Ray(p + normal * RT_EPS, normalize(dir));
            attenuation = mat.albedo;
        }
        return true;
    }

    if (mat.type == MAT_BUILDING_FACADE) {
        float u = p.x + p.z;
        float v = p.y;
        float cu = floorf(u * 1.35f);
        float cv = floorf(v * 0.95f);
        float fu = u * 1.35f - cu;
        float fv = v * 0.95f - cv;

        // Exclude lower street ground level from grid & keep dark mullions between floors
        bool in_pane = (v > 2.2f) && (fu > 0.16f && fu < 0.84f && fv > 0.22f && fv < 0.78f);
        float h = mat_hash2(static_cast<int>(cu), static_cast<int>(cv));
        if (in_pane && h < mat.param) {
            float var = 0.45f + 0.55f * mat_hash2(static_cast<int>(cu) + 19, static_cast<int>(cv) + 41);
            Vec3 wcol = mat.albedo2;
            float hue_pick = mat_hash2(static_cast<int>(cu) * 3 + 7, static_cast<int>(cv) * 5 + 11);
            if (hue_pick > 0.78f) wcol = Vec3(1.0f, 0.62f, 0.28f); // warm office window
            else if (hue_pick < 0.18f) wcol = Vec3(0.95f, 0.25f, 0.78f); // neon magenta room
            emitted = wcol * (mat.emission.x * var);
            attenuation = Vec3(0.0f);
            return false;
        }

        Vec3 unit_dir = normalize(r_in.dir);
        if (rng.next_float() < 0.35f) {
            Vec3 refl = reflect(unit_dir, normal) + 0.24f * rng.random_unit_vector();
            if (dot(refl, normal) <= 0.0f) refl = normal + rng.random_unit_vector();
            scattered = Ray(p + normal * RT_EPS, normalize(refl));
        } else {
            Vec3 sd = normal + rng.random_unit_vector();
            if (sd.near_zero()) sd = normal;
            scattered = Ray(p + normal * RT_EPS, normalize(sd));
        }
        attenuation = mat.albedo;
        return true;
    }

    if (mat.type == MAT_NEON_SIGN) {
        float sc = mat.param > 0.0f ? mat.param : 2.4f;
        float su = (p.x * 0.85f + p.z * 1.15f) * sc;
        float sv = p.y * sc;
        int gx = static_cast<int>(floorf(su));
        int gy = static_cast<int>(floorf(sv));
        float lx = su - static_cast<float>(gx);
        float ly = sv - static_cast<float>(gy);

        uint32_t code = static_cast<uint32_t>(mat_hash2(gx, gy) * 63.0f);
        bool stroke = false;
        if (lx > 0.14f && lx < 0.86f && ly > 0.14f && ly < 0.86f) {
            if ((code & 1u)  && fabsf(ly - 0.76f) < 0.09f) stroke = true;
            if ((code & 2u)  && fabsf(ly - 0.50f) < 0.09f) stroke = true;
            if ((code & 4u)  && fabsf(ly - 0.24f) < 0.09f) stroke = true;
            if ((code & 8u)  && fabsf(lx - 0.26f) < 0.09f) stroke = true;
            if ((code & 16u) && fabsf(lx - 0.74f) < 0.09f) stroke = true;
            if ((code & 32u) && fabsf(lx - 0.50f) < 0.09f) stroke = true;
        }
        // Subtle glowing cell edge accent
        if (!stroke && (lx < 0.05f || lx > 0.95f || ly < 0.05f || ly > 0.95f)) {
            if (mat_hash2(gx + 3, gy + 7) > 0.45f) stroke = true;
        }

        if (stroke) {
            emitted = mat.albedo * mat.emission.x;
            attenuation = Vec3(0.0f);
            return false;
        }
        Vec3 sd = normal + rng.random_unit_vector();
        if (sd.near_zero()) sd = normal;
        scattered = Ray(p + normal * RT_EPS, normalize(sd));
        attenuation = mat.albedo * 0.05f;
        return true;
    }

    if (mat.type == MAT_METAL) {
        Vec3 unit_dir = normalize(r_in.dir);
        Vec3 reflected = reflect(unit_dir, normal);
        Vec3 dir = reflected + mat.param * rng.random_unit_vector();
        attenuation = mat.albedo;
        if (dot(dir, normal) > 0.0f) {
            scattered = Ray(p + normal * RT_EPS, normalize(dir));
            return true;
        }
        return false;
    }

    if (mat.type == MAT_GLASS) {
        attenuation = mat.albedo;
        float refraction_ratio = front_face ? (1.0f / mat.param) : mat.param;
        Vec3 unit_direction = normalize(r_in.dir);
        float cos_theta = rt_min(dot(-unit_direction, normal), 1.0f);
        float sin_theta = sqrtf(rt_max(0.0f, 1.0f - cos_theta * cos_theta));

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        Vec3 direction;
        Vec3 offset_normal;
        if (cannot_refract || schlick(cos_theta, refraction_ratio) > rng.next_float()) {
            direction = reflect(unit_direction, normal);
            offset_normal = normal;
        } else {
            direction = refract(unit_direction, normal, refraction_ratio);
            offset_normal = -normal;
        }
        scattered = Ray(p + offset_normal * RT_EPS, normalize(direction));
        return true;
    }

    return false;
}
