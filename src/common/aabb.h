#pragma once

#include "ray.h"

struct AABB {
    Vec3 bmin;
    Vec3 bmax;

    RT_HD AABB()
        : bmin(Vec3(RT_INF, RT_INF, RT_INF)),
          bmax(Vec3(-RT_INF, -RT_INF, -RT_INF)) {}

    RT_HD AABB(const Vec3& mn, const Vec3& mx) : bmin(mn), bmax(mx) {}

    RT_HD static AABB empty() { return AABB(); }

    RT_HD void expand(const Vec3& p) {
        bmin.x = fminf(bmin.x, p.x);
        bmin.y = fminf(bmin.y, p.y);
        bmin.z = fminf(bmin.z, p.z);
        bmax.x = fmaxf(bmax.x, p.x);
        bmax.y = fmaxf(bmax.y, p.y);
        bmax.z = fmaxf(bmax.z, p.z);
    }

    RT_HD void expand(const AABB& other) {
        expand(other.bmin);
        expand(other.bmax);
    }

    RT_HD static AABB surrounding(const AABB& a, const AABB& b) {
        AABB r = a;
        r.expand(b);
        return r;
    }

    RT_HD Vec3 centroid() const {
        return 0.5f * (bmin + bmax);
    }

    RT_HD int longest_axis() const {
        float dx = bmax.x - bmin.x;
        float dy = bmax.y - bmin.y;
        float dz = bmax.z - bmin.z;
        if (dx >= dy && dx >= dz) return 0;
        if (dy >= dz) return 1;
        return 2;
    }

    // IEEE-754 fminf/fmaxf slab intersection (safe with +/-Inf, zero -ffast-math)
    RT_HD bool hit(const Ray& r, const Vec3& inv_dir, float t_min, float t_max, float& out_tmin) const {
        float tx0 = (bmin.x - r.orig.x) * inv_dir.x;
        float tx1 = (bmax.x - r.orig.x) * inv_dir.x;
        float tmin = fmaxf(t_min, fminf(tx0, tx1));
        float tmax = fminf(t_max, fmaxf(tx0, tx1));

        float ty0 = (bmin.y - r.orig.y) * inv_dir.y;
        float ty1 = (bmax.y - r.orig.y) * inv_dir.y;
        tmin = fmaxf(tmin, fminf(ty0, ty1));
        tmax = fminf(tmax, fmaxf(ty0, ty1));

        float tz0 = (bmin.z - r.orig.z) * inv_dir.z;
        float tz1 = (bmax.z - r.orig.z) * inv_dir.z;
        tmin = fmaxf(tmin, fminf(tz0, tz1));
        tmax = fminf(tmax, fmaxf(tz0, tz1));

        if (tmax < tmin) return false;
        out_tmin = tmin;
        return true;
    }
};
