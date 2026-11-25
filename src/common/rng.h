#pragma once

#include "vec3.h"

RT_HD RT_INLINE uint32_t pcg_hash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct Rng {
    uint32_t state;

    RT_HD Rng() : state(1u) {}
    RT_HD explicit Rng(uint32_t seed) {
        state = pcg_hash(seed ^ 0x9E3779B9u) | 1u;
    }

    RT_HD static Rng from_pixel(uint32_t seed, int x, int y, int sample_idx) {
        uint32_t h = pcg_hash(seed ^ 0x9E3779B9u);
        h = pcg_hash(h ^ (static_cast<uint32_t>(x) * 1664525u + 1013904223u));
        h = pcg_hash(h ^ (static_cast<uint32_t>(y) * 22695477u + 1u));
        h = pcg_hash(h ^ (static_cast<uint32_t>(sample_idx) * 747796405u + 2891336453u));
        Rng r;
        r.state = h | 1u;
        return r;
    }

    RT_HD uint32_t next_uint() {
        state = state * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    RT_HD float next_float() {
        return static_cast<float>(next_uint() >> 8u) * (1.0f / 16777216.0f);
    }

    RT_HD float next_float(float lo, float hi) {
        return lo + (hi - lo) * next_float();
    }

    // Analytical uniform spherical sampling (zero rejection-loop warp divergence)
    RT_HD Vec3 random_unit_vector() {
        float z = 2.0f * next_float() - 1.0f;
        float a = 2.0f * RT_PI * next_float();
        float r = sqrtf(rt_max(0.0f, 1.0f - z * z));
        return Vec3(r * cosf(a), r * sinf(a), z);
    }

    RT_HD Vec3 random_in_unit_disk() {
        float r = sqrtf(next_float());
        float theta = 2.0f * RT_PI * next_float();
        return Vec3(r * cosf(theta), r * sinf(theta), 0.0f);
    }
};
