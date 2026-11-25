#pragma once

#include "primitive.h"
#include <algorithm>
#include <vector>

struct BVHNode {
    AABB box;
    int  left;
    int  right;
    int  first;
    int  count;
};

namespace detail {
inline int build_bvh_recursive(
    std::vector<Primitive>& prims,
    std::vector<BVHNode>& nodes,
    int start,
    int end
) {
    int node_idx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    AABB total_box = AABB::empty();
    AABB centroid_box = AABB::empty();
    for (int i = start; i < end; ++i) {
        AABB b = primitive_bounds(prims[i]);
        total_box.expand(b);
        centroid_box.expand(b.centroid());
    }

    int count = end - start;
    if (count <= 4) {
        nodes[node_idx].box = total_box;
        nodes[node_idx].left = -1;
        nodes[node_idx].right = -1;
        nodes[node_idx].first = start;
        nodes[node_idx].count = count;
        return node_idx;
    }

    int axis = centroid_box.longest_axis();
    float mid_coord = 0.5f * (centroid_box.bmin[axis] + centroid_box.bmax[axis]);

    auto mid_it = std::partition(
        prims.begin() + start,
        prims.begin() + end,
        [axis, mid_coord](const Primitive& p) {
            return primitive_bounds(p).centroid()[axis] < mid_coord;
        }
    );

    int mid = static_cast<int>(mid_it - prims.begin());
    if (mid == start || mid == end) {
        mid = start + count / 2;
        std::nth_element(
            prims.begin() + start,
            prims.begin() + mid,
            prims.begin() + end,
            [axis](const Primitive& a, const Primitive& b) {
                return primitive_bounds(a).centroid()[axis] <
                       primitive_bounds(b).centroid()[axis];
            }
        );
    }

    int left_child = build_bvh_recursive(prims, nodes, start, mid);
    int right_child = build_bvh_recursive(prims, nodes, mid, end);

    nodes[node_idx].box = total_box;
    nodes[node_idx].left = left_child;
    nodes[node_idx].right = right_child;
    nodes[node_idx].first = -1;
    nodes[node_idx].count = 0;
    return node_idx;
}
} // namespace detail

inline void build_bvh(std::vector<Primitive>& prims, std::vector<BVHNode>& nodes) {
    nodes.clear();
    if (prims.empty()) return;
    nodes.reserve(prims.size() * 2);
    detail::build_bvh_recursive(prims, nodes, 0, static_cast<int>(prims.size()));
}

// Iterative BVH traversal with fixed local stack (zero recursion for CPU/CUDA parity)
RT_HD RT_INLINE bool bvh_intersect(
    const BVHNode* nodes,
    int node_count,
    const Primitive* prims,
    const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec
) {
    if (node_count <= 0) return false;

    Vec3 inv_dir(1.0f / r.dir.x, 1.0f / r.dir.y, 1.0f / r.dir.z);
    float root_tmin = 0.0f;
    if (!nodes[0].box.hit(r, inv_dir, t_min, t_max, root_tmin)) {
        return false;
    }

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    bool hit_anything = false;
    float closest = t_max;

    while (sp > 0) {
        int idx = stack[--sp];
        const BVHNode& node = nodes[idx];

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                HitRecord temp;
                if (intersect_primitive(prims[node.first + i], r, t_min, closest, temp)) {
                    hit_anything = true;
                    closest = temp.t;
                    rec = temp;
                }
            }
            continue;
        }

        float tmin_l = RT_INF;
        float tmin_r = RT_INF;
        bool hit_l = (node.left >= 0) && nodes[node.left].box.hit(r, inv_dir, t_min, closest, tmin_l);
        bool hit_r = (node.right >= 0) && nodes[node.right].box.hit(r, inv_dir, t_min, closest, tmin_r);

        if (hit_l && hit_r) {
            // Push farther child first so closer child is popped first
            if (tmin_l < tmin_r) {
                if (sp < 64) stack[sp++] = node.right;
                if (sp < 64) stack[sp++] = node.left;
            } else {
                if (sp < 64) stack[sp++] = node.left;
                if (sp < 64) stack[sp++] = node.right;
            }
        } else if (hit_l) {
            if (sp < 64) stack[sp++] = node.left;
        } else if (hit_r) {
            if (sp < 64) stack[sp++] = node.right;
        }
    }

    return hit_anything;
}
