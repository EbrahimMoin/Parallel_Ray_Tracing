#pragma once

#include <cmath>
#include <cstdint>
#include <cfloat>

#ifdef __CUDACC__
#define RT_HD __host__ __device__
#define RT_INLINE __forceinline__
#else
#define RT_HD
#define RT_INLINE inline
#endif

constexpr float RT_PI  = 3.14159265358979323846f;
constexpr float RT_INF = 1e30f;
constexpr float RT_EPS = 1e-4f;

RT_HD RT_INLINE float rt_clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

RT_HD RT_INLINE float rt_min(float a, float b) {
    return a < b ? a : b;
}

RT_HD RT_INLINE float rt_max(float a, float b) {
    return a > b ? a : b;
}

RT_HD RT_INLINE float rt_lerp(float a, float b, float t) {
    return a + t * (b - a);
}
