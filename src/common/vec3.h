#pragma once

#include "rt_common.h"

struct Vec3 {
    float x, y, z;

    RT_HD Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    RT_HD explicit Vec3(float v) : x(v), y(v), z(v) {}
    RT_HD Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    RT_HD Vec3 operator-() const { return Vec3(-x, -y, -z); }

    RT_HD float operator[](int i) const {
        return i == 0 ? x : (i == 1 ? y : z);
    }
    RT_HD float& operator[](int i) {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    RT_HD Vec3& operator+=(const Vec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    RT_HD Vec3& operator*=(float t) {
        x *= t; y *= t; z *= t;
        return *this;
    }

    RT_HD Vec3& operator*=(const Vec3& v) {
        x *= v.x; y *= v.y; z *= v.z;
        return *this;
    }

    RT_HD Vec3& operator/=(float t) {
        float inv = 1.0f / t;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }

    RT_HD float length_squared() const {
        return x * x + y * y + z * z;
    }

    RT_HD float length() const {
        return sqrtf(length_squared());
    }

    RT_HD bool near_zero() const {
        const float s = 1e-6f;
        return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
    }
};

RT_HD RT_INLINE Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

RT_HD RT_INLINE Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

RT_HD RT_INLINE Vec3 operator*(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

RT_HD RT_INLINE Vec3 operator*(const Vec3& v, float t) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

RT_HD RT_INLINE Vec3 operator*(float t, const Vec3& v) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

RT_HD RT_INLINE Vec3 operator/(const Vec3& v, float t) {
    float inv = 1.0f / t;
    return Vec3(v.x * inv, v.y * inv, v.z * inv);
}

RT_HD RT_INLINE float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

RT_HD RT_INLINE Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

RT_HD RT_INLINE Vec3 normalize(const Vec3& v) {
    float len = v.length();
    if (len <= 1e-12f) return Vec3(0.0f, 1.0f, 0.0f);
    return v / len;
}

RT_HD RT_INLINE Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

RT_HD RT_INLINE Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = rt_min(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float perp_sq = r_out_perp.length_squared();
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - perp_sq)) * n;
    return r_out_perp + r_out_parallel;
}

RT_HD RT_INLINE float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    float one_minus = 1.0f - cosine;
    float p2 = one_minus * one_minus;
    return r0 + (1.0f - r0) * (p2 * p2 * one_minus);
}
