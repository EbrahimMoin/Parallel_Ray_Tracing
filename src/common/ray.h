#pragma once

#include "vec3.h"

struct Ray {
    Vec3 orig;
    Vec3 dir;

    RT_HD Ray() = default;
    RT_HD Ray(const Vec3& o, const Vec3& d) : orig(o), dir(d) {}

    RT_HD Vec3 at(float t) const {
        return orig + t * dir;
    }
};
