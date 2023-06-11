#include "World.h"
#include "File.h"
#include <stdexcept>
#include <tuple>
#include <cmath>
#include <unordered_map>

namespace {
    World::Box null_box() {
        return World::Box{
            Vec3{HUGE_VALF, HUGE_VALF, HUGE_VALF},
            Vec3{-HUGE_VALF, -HUGE_VALF, -HUGE_VALF},
        };
    }

    void box_union_with(World::Box& a, World::Box const& b) {
        a.min.x = fminf(a.min.x, b.min.x);
        a.min.y = fminf(a.min.y, b.min.y);
        a.min.z = fminf(a.min.z, b.min.z);
        a.max.x = fmaxf(a.max.x, b.max.x);
        a.max.y = fmaxf(a.max.y, b.max.y);
        a.max.z = fmaxf(a.max.z, b.max.z);
    }

    World::Box vec3_box(Vec3 const& a) {
        return World::Box{a, a};
    }

    World::Box triangle_box(Vec3 a, Vec3 b, Vec3 c) {
        World::Box box = vec3_box(a);
        box_union_with(box, vec3_box(b));
        box_union_with(box, vec3_box(c));
        return box;
    }

    struct WorldLoader {
        World world;
        File f;
        
        void load_magic() {
            auto magic = f.read<uint64_t>();
            if (magic != 0xdb7652c5014bac40ULL) {
                throw std::runtime_error("Invalid file format");
            }
        }
        
        void load_strings() {
            auto string_count = f.read<int32_t>();
            world.material_names.resize(string_count);
            for (int i = 0; i < string_count; ++i) {
                int name_size = f.read<int32_t>();
                std::string& name = world.material_names[i];
                name.resize(name_size, 0);
                f.read_bytes(name_size, name.data());
            }
        }
        
        void load_surfaces() {
            auto surface_count = f.read<int32_t>();
            world.surfaces.resize(surface_count);
            for (int i = 0; i < surface_count; ++i) {
                World::Surface& surface = world.surfaces[i];
                surface.material_index = f.read<int32_t>();
                if (surface.material_index < 0 || surface.material_index >= world.material_names.size()) {
                    throw std::runtime_error("invalid material index");
                }
                surface.flags = f.read<uint32_t>();
                surface.texture_space = f.read<Xform3>();
            }
        }
        
        void load_vertices() {
            auto vertex_count = f.read<int32_t>();
            world.vertices.resize(vertex_count);
            for (int i = 0; i < vertex_count; ++i) {
                world.vertices[i] = f.read<Vec3>();
            }
        }
        
        void load_normals() {
            auto normal_count = f.read<int32_t>();
            world.normals.resize(normal_count);
            for (int i = 0; i < normal_count; ++i) {
                world.normals[i] = f.read<Vec3>();
            }
        }
        
        void read_vertex(int32_t& index, Vec3& vertex) {
            index = f.read<int32_t>();
            if (index < 0 || index  >= world.vertices.size()) {
                throw std::runtime_error("invalid vertex index");
            }
            vertex = world.vertices[index];
        }
        
        void read_normal_index(int32_t& index) {
            index = f.read<int32_t>();
            if (index < 0 || index  >= world.normals.size()) {
                throw std::runtime_error("invalid normal index");
            }
        }

        void read_surface_index(int32_t& index) {
            index = f.read<int32_t>();
            if (index < 0 || index >= world.surfaces.size()) {
                throw std::runtime_error("invalid surface index");
            }
        }

        void read_zone_index(int32_t& index) {
            index = f.read<int32_t>();
            if (index < 0 || index >= world.zone_trees.size()) {
                throw std::runtime_error("invalid zone index");
            }
        }
        
        World::Box load_tree_node_leaf(World::Tree& tree) {
            World::Triangle& tri = tree.triangles.emplace_back();
            Vec3 a, b, c;
            read_vertex(tri.vertex_indexes[0], a);
            read_vertex(tri.vertex_indexes[1], b);
            read_vertex(tri.vertex_indexes[2], c);
            read_normal_index(tri.normal_index);
            read_surface_index(tri.surface_index);
            read_zone_index(tri.other_zone_index);
            World::Box tri_box = triangle_box(a, b, c);
            tree.node_boxes[0].push_back(tri_box);
            return tri_box;
        }
        
        World::Box load_tree_node_branch(World::Tree& tree, int level) {
            size_t children_count = f.read<int32_t>();
            World::Box node_box = null_box();
            for (int i = 0; i < children_count; ++i) {
                auto child_level = f.read<int32_t>();
                if (child_level != level - 1) {
                    throw std::runtime_error("Invalid tree structure");
                }
                World::Box child_box = load_tree_node(tree, child_level);
                box_union_with(node_box, child_box);
            }
            tree.node_boxes[level].push_back(node_box);
            tree.node_children_indexes[level - 1].push_back(tree.node_boxes[level - 1].size());
            return node_box;
        }
        
        World::Box load_tree_node(World::Tree& tree, int level) {
            if (level == 0) {
                return load_tree_node_leaf(tree);
            } else {
                return load_tree_node_branch(tree, level);
            }
        }

        World::Tree load_triangle_tree() {
            World::Tree tree;
            auto root_level = f.read<int32_t>();
            tree.node_boxes.resize(root_level + 1);
            tree.node_children_indexes.resize(root_level);
            for (size_t i = 0; i < root_level; ++i) {
                tree.node_children_indexes[i].push_back(0);
            }
            load_tree_node(tree, root_level);
            tree.node_boxes[0].shrink_to_fit();
            for (size_t i = 0; i < root_level; ++i) {
                tree.node_boxes[i + 1].shrink_to_fit();
                tree.node_children_indexes[i].shrink_to_fit();
            }
            return tree;
        }

        void load_zones() {
            auto zone_count = f.read<int32_t>();
            world.zone_trees.resize(zone_count);
            for (size_t i = 0; i < zone_count; ++i) {
                auto zone_flags = f.read<uint32_t>();
                world.zone_trees[i] = load_triangle_tree();
                if (zone_flags & 1) {
                    world.zone_at_infinity = i;
                }
            }
        }
    };

    void xray_intersection(World const& world, float& hit_x, Vec3 pos, World::Tree const& tree, size_t node_level, size_t node_index) {
        World::Box const& box = tree.node_boxes[node_level][node_index];
        if (
            box.min.x < hit_x &&
            pos.x <= box.max.x &&
            box.min.y <= pos.y && pos.y <= box.max.y &&
            box.min.z <= pos.z && pos.z <= box.max.z
        ) {
            if (node_level == 0) {
                World::Triangle const& tri = tree.triangles[node_index];
                Vec3 a = world.vertices[tri.vertex_indexes[0]] - pos;
                Vec3 b = world.vertices[tri.vertex_indexes[1]] - pos;
                Vec3 c = world.vertices[tri.vertex_indexes[2]] - pos;
                float u = c.y * b.z - c.z * b.y;
                float v = a.y * c.z - a.z * c.y;
                float w = b.y * a.z - b.z * a.y;
                if (u < 0 || v < 0 || w < 0) {
                    return;
                }
                float det = u + v + w;
                if (det == 0) {
                    return;
                }
                float t = u * a.x + v * b.x + w * c.x;
                float x = t / det;
                if (pos.x <= x && x < hit_x) {
                    hit_x = x;
                }
            } else {
                size_t children_begin = tree.node_children_indexes[node_level - 1][node_index];
                size_t children_end = tree.node_children_indexes[node_level - 1][node_index + 1];
                for (size_t i = children_begin; i < children_end; ++i) {
                    xray_intersection(world, hit_x, pos, tree, node_level - 1, i);
                }
            }
        }
    }
}

size_t World::zone_index_at(Vec3 pos) const {
    size_t current_zone = zone_at_infinity;
    float current_x = HUGE_VALF;
    for (size_t i = 0; i < zone_trees.size(); ++i) {
        World::Tree const& tree = zone_trees[i];
        float hit_x = current_x;
        xray_intersection(*this, hit_x, pos, tree, tree.node_boxes.size() - 1, 0);
        if (hit_x < current_x) {
            current_zone = i;
            current_x = hit_x;
        }
    }
    return current_zone;
}

World load_world(std::string const& path) {
    WorldLoader loader{
        World{},
        File(path, "rb"),
    };
    loader.load_magic();
    loader.load_strings();
    loader.load_surfaces();
    loader.load_vertices();
    loader.load_normals();
    loader.load_zones();
    return loader.world;
}

World load_test_world() {
    World world;
    world.material_names = std::vector{
        std::string("a"),
    };
    world.surfaces = std::vector{
        World::Surface{0, 0, Xform3{}},
    };
    world.vertices = std::vector{
        Vec3{-100, 0, -100},
        Vec3{100, 0, -100},
        Vec3{100, 0,  100},
        Vec3{-100, 0,  100},
    };
    world.zone_trees.resize(1);
    world.zone_trees[0].triangles = std::vector{
        World::Triangle{{0, 1, 2}, 0},
        World::Triangle{{2, 1, 0}, 0},
        World::Triangle{{0, 2, 3}, 0},
        World::Triangle{{3, 2, 0}, 0},
    };
    world.zone_trees[0].node_boxes.resize(2);
    world.zone_trees[0].node_boxes[0] = std::vector{
        World::Box{{-100, -50, -100}, {100, 50, 100}},
        World::Box{{-100, -50, -100}, {100, 50, 100}},
        World::Box{{-100, -50, -100}, {100, 50, 100}},
        World::Box{{-100, -50, -100}, {100, 50, 100}},
    };
    world.zone_trees[0].node_boxes[1] = std::vector{
        World::Box{{-100, -50, -100}, {100, 50, 100}},
    };
    world.zone_trees[0].node_children_indexes.resize(1);
    world.zone_trees[0].node_children_indexes[0] = std::vector{
        (size_t)0,
        (size_t)4,
    };
    return world;
}
