#pragma once

#include "Vec3.h"
#include <string>
#include <vector>
#include <math.h>

struct World {
    static constexpr uint32_t flag_masked = 1;
    static constexpr uint32_t flag_translucent = 2;
    static constexpr uint32_t flag_modulated = 4;
    static constexpr uint32_t flag_unlit = 8;
    static constexpr uint32_t flag_mirror = 16;
    static constexpr uint32_t flag_invisible = 32;
    static constexpr size_t invalid_index = (size_t)(-1);

    struct Triangle {
        int vertex_indexes[3];
        int normal_index;
        int surface_index;
        int other_zone_id;
        int zone_id;
        float area;
    };

    struct Box {
        Vec3 min, max;
    };

    struct Surface {
        int material_index;
        uint32_t flags;
        Xform3 texture_space;
    };

    struct Tree {
        size_t triangle_index_offset;
        std::vector<std::vector<Box>> node_boxes;
        std::vector<std::vector<size_t>> node_children_indexes;
    };

    struct NodeId {
        size_t level, index;
    };

    std::vector<std::string> material_names;
    std::vector<Surface> surfaces;
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<Triangle> triangles;
    std::vector<Tree> zone_trees;
    size_t zone_at_infinity;

    size_t zone_id_at(Vec3 pos) const;
};

World load_world(std::string const& path);
World load_test_world();
