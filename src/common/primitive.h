#pragma once

#include "aabb.h"

enum PrimType {
    PRIM_SPHERE   = 0,
    PRIM_TRIANGLE = 1
};

struct Primitive {
    int   type;
    int   mat_id;
    Vec3  v0;     // sphere center, or triangle vertex A
    Vec3  v1;     // triangle edge1 (B - A)
    Vec3  v2;     // triangle edge2 (C - A)
    Vec3  normal; // precomputed normalized geometric triangle normal
    Vec3  n0;     // vertex A normal
    Vec3  n1;     // vertex B normal
    Vec3  n2;     // vertex C normal
    float radius; // sphere radius

    RT_HD Primitive()
        : type(PRIM_SPHERE), mat_id(0), radius(1.0f) {}

    RT_HD static Primitive make_sphere(const Vec3& center, float r, int mat) {
        Primitive p;
        p.type = PRIM_SPHERE;
        p.mat_id = mat;
        p.v0 = center;
        p.radius = r;
        return p;
    }

    RT_HD static Primitive make_triangle(const Vec3& a, const Vec3& b, const Vec3& c, int mat) {
        Primitive p;
        p.type = PRIM_TRIANGLE;
        p.mat_id = mat;
        p.v0 = a;
        p.v1 = b - a;
        p.v2 = c - a;
        p.normal = normalize(cross(p.v1, p.v2));
        p.n0 = p.normal;
        p.n1 = p.normal;
        p.n2 = p.normal;
        p.radius = 0.0f;
        return p;
    }

    RT_HD static Primitive make_smooth_triangle(
        const Vec3& a, const Vec3& b, const Vec3& c,
        const Vec3& na, const Vec3& nb, const Vec3& nc,
        int mat
    ) {
        Primitive p;
        p.type = PRIM_TRIANGLE;
        p.mat_id = mat;
        p.v0 = a;
        p.v1 = b - a;
        p.v2 = c - a;
        p.normal = normalize(cross(p.v1, p.v2));
        p.n0 = normalize(na);
        p.n1 = normalize(nb);
        p.n2 = normalize(nc);
        p.radius = 0.0f;
        return p;
    }
};

struct HitRecord {
    float t;
    Vec3  p;
    Vec3  normal;
    int   mat_id;
    bool  front_face;

    RT_HD void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.dir, outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

RT_HD RT_INLINE AABB primitive_bounds(const Primitive& p) {
    if (p.type == PRIM_SPHERE) {
        Vec3 rvec(p.radius, p.radius, p.radius);
        return AABB(p.v0 - rvec, p.v0 + rvec);
    }
    Vec3 a = p.v0;
    Vec3 b = p.v0 + p.v1;
    Vec3 c = p.v0 + p.v2;
    AABB box = AABB::empty();
    box.expand(a);
    box.expand(b);
    box.expand(c);
    Vec3 pad(1e-4f, 1e-4f, 1e-4f);
    box.bmin = box.bmin - pad;
    box.bmax = box.bmax + pad;
    return box;
}

RT_HD RT_INLINE bool intersect_primitive(
    const Primitive& p,
    const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec
) {
    if (p.type == PRIM_SPHERE) {
        Vec3 oc = r.orig - p.v0;
        float a = dot(r.dir, r.dir);
        float half_b = dot(oc, r.dir);
        float c = dot(oc, oc) - p.radius * p.radius;
        float disc = half_b * half_b - a * c;
        if (disc < 0.0f) return false;
        float sqrtd = sqrtf(disc);

        float root = (-half_b - sqrtd) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max) return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        Vec3 outward = (rec.p - p.v0) / p.radius;
        rec.set_face_normal(r, outward);
        rec.mat_id = p.mat_id;
        return true;
    }

    // Moller-Trumbore ray-triangle intersection with barycentric smooth normal
    Vec3 h = cross(r.dir, p.v2);
    float a = dot(p.v1, h);
    if (fabsf(a) < 1e-7f) return false;
    float f = 1.0f / a;
    Vec3 s = r.orig - p.v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    Vec3 q = cross(s, p.v1);
    float v = f * dot(r.dir, q);
    if (v < 0.0f || (u + v) > 1.0f) return false;
    float t = f * dot(p.v2, q);
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.at(t);
    float w = 1.0f - u - v;
    Vec3 interp_n = normalize(w * p.n0 + u * p.n1 + v * p.n2);
    rec.set_face_normal(r, interp_n);
    rec.mat_id = p.mat_id;
    return true;
}
