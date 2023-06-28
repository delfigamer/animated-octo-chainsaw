#include "RayPipeline.h"
#include "Ymm.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <immintrin.h>
        
struct AlignedBufferLayout {
    struct Item {
        void const* source_ptr;
        void** target_ptr_ptr;
        size_t begin;
        size_t end;
    };

    std::vector<Item> items;

    void add_item(void const* source_ptr, void** target_ptr_ptr, size_t elem_count, size_t elem_size, size_t elem_align) {
        size_t begin = total_size();
        begin += elem_align - 1;
        begin -= begin % elem_align;
        size_t end = begin + elem_count * elem_size;
        items.push_back(Item{source_ptr, target_ptr_ptr, begin, end});
    }

    size_t total_size() {
        if (items.empty()) {
            return 0;
        } else {
            return items.back().end;
        }
    }

    void transfer_data(char* target_ptr) {
        for (Item const& item : items) {
            if (item.source_ptr) {
                memcpy(target_ptr + item.begin, item.source_ptr, item.end - item.begin);
            }
            *item.target_ptr_ptr = target_ptr + item.begin;
        }
    }

    template<typename T, typename F>
    void add_item_vector(std::vector<T> const& vector, F*& target_ptr) {
        add_item(vector.data(), (void**)&target_ptr, vector.size(), sizeof(T), alignof(T));
    }
};

struct RayPipeline::Internal {
    struct Loader;
    struct Utils;
    struct RaySpaceShear;
    struct ChunkIntersectorImmediate;
    struct ChunkIntersectorSimd;
    struct ChunkIntersectorBulk;
    struct RayScheduler;
    struct ChunkProcessor;
    struct TreeProcessor;

    static void append_ray(RayPipeline& pipeline, size_t zone_id, std::unique_ptr<RayPacket>& packet_ptr, RayPacketFunnel& target_funnel, RayData ray_data);
};

size_t RayPipelineParams::sizeof_packed_triangle() const {
    switch (triangle_rep) {
    case RayPipelineParams::TriangleRep::TriplePointIndexed:
        return sizeof(RayPipeline::PackedTriangleTriplePointIndexed);
    case RayPipelineParams::TriangleRep::TriplePointImmediate:
        return sizeof(RayPipeline::PackedTriangleTriplePointImmediate);
    case RayPipelineParams::TriangleRep::BoxCenteredUV:
        return sizeof(RayPipeline::PackedTriangleBoxCenteredUV);
    case RayPipelineParams::TriangleRep::BoxCenteredUVExplicitN:
        return sizeof(RayPipeline::PackedTriangleBoxCenteredUVExplicitN);
    case RayPipelineParams::TriangleRep::MiddleCoordPermuted:
        return sizeof(RayPipeline::PackedTriangleMiddleCoordPermuted);
    default:
        throw std::runtime_error("invalid triangle representation value");
    }
}

size_t RayPipelineParams::alignof_packed_triangle() const {
    switch (triangle_rep) {
    case RayPipelineParams::TriangleRep::TriplePointIndexed:
        return alignof(RayPipeline::PackedTriangleTriplePointIndexed);
    case RayPipelineParams::TriangleRep::TriplePointImmediate:
        return alignof(RayPipeline::PackedTriangleTriplePointImmediate);
    case RayPipelineParams::TriangleRep::BoxCenteredUV:
        return alignof(RayPipeline::PackedTriangleBoxCenteredUV);
    case RayPipelineParams::TriangleRep::BoxCenteredUVExplicitN:
        return alignof(RayPipeline::PackedTriangleBoxCenteredUVExplicitN);
    case RayPipelineParams::TriangleRep::MiddleCoordPermuted:
        return alignof(RayPipeline::PackedTriangleMiddleCoordPermuted);
    default:
        throw std::runtime_error("invalid triangle representation value");
    }
}

bool Ray::NodeReservation::compare_node_reservation_greater(NodeReservation const& a, NodeReservation const& b) {
    return a.min_hit_param > b.min_hit_param;
}

Ray::SpaceShear Ray::SpaceShear::make_for(Vec3 dir) {
    int max_dim = 0;
    if (fabsf(dir[1]) > fabsf(dir[0])) {
        max_dim = 1;
    }
    if (fabsf(dir[2]) > fabsf(dir[max_dim])) {
        max_dim = 2;
    }
    SpaceShear rss;
    rss.kz = max_dim;
    rss.kx = (rss.kz + 1) % 3;
    rss.ky = (rss.kx + 1) % 3;
    if (dir[rss.kz] < 0) {
        std::swap(rss.kx, rss.ky);
    }
    rss.sx = dir[rss.kx] / dir[rss.kz];
    rss.sy = dir[rss.ky] / dir[rss.kz];
    rss.sz = 1.0f / dir[rss.kz];
    return rss;
}
/*
    1 2 3       1 3 2           0 1     0
    2 1 3       3 1 2           2 3     1
    2 3 1       3 2 1           4 5     2
*/

Ray::Ray(size_t zone_index, Vec3 origin, Vec3 direction, float min_param, float max_param) {
    reset(zone_index, origin, direction, min_param, max_param);
}

Ray::~Ray() {
}

Ray::NodeReservation Ray::pop_min_reservation() {
    std::pop_heap(
        _reservation_min_heap.begin(),
        _reservation_min_heap.end(),
        NodeReservation::compare_node_reservation_greater);
    NodeReservation min_res = _reservation_min_heap.back();
    _reservation_min_heap.pop_back();
    return min_res;
}

void Ray::insert_reservation(size_t node_index, float min_hit_param) {
    _reservation_min_heap.push_back(NodeReservation{node_index, min_hit_param});
    std::push_heap(
        _reservation_min_heap.begin(),
        _reservation_min_heap.end(),
        NodeReservation::compare_node_reservation_greater);
}

void Ray::reset(size_t zone_index, float min_param, float max_param) {
    reset(zone_index, hit_point(), _direction, min_param, max_param);
}

void Ray::reset(Vec3 direction, float min_param, float max_param) {
    reset(hit_point(), direction, min_param, max_param);
}

void Ray::reset(size_t zone_index, Vec3 direction, float min_param, float max_param) {
    reset(zone_index, hit_point(), direction, min_param, max_param);
}

void Ray::reset(Vec3 origin, Vec3 direction, float min_param, float max_param) {
    reset(_zone_index, origin, direction, min_param, max_param);
}

void Ray::reset(size_t zone_index, Vec3 origin, Vec3 direction, float min_param, float max_param) {
    _zone_index = zone_index;
    _origin = origin;
    _direction = direction;
    _dir_inverse = elementwise_inverse(direction);
    _space_shear = SpaceShear::make_for(_direction);
    _min_param = min_param;
    _max_param = max_param;
    _hit_triangle_index = World::invalid_index;
    _hit_triangle_pos = Vec3{0, 0, 0};
    _reservation_min_heap.clear();
}

void Ray::on_completed(std::unique_ptr<Ray>& self_ptr, RayPipeline& pipeline) {
}

bool RayPipeline::ReservationHeap::NodeReservation::compare_greater(NodeReservation const& a, NodeReservation const& b) {
    return a.min_hit_param > b.min_hit_param;
}

RayPipeline::ReservationHeap::NodeReservation RayPipeline::ReservationHeap::pop_min_reservation() {
    std::pop_heap(heap_data.begin(), heap_data.end(), NodeReservation::compare_greater);
    NodeReservation min_res = heap_data.back();
    heap_data.pop_back();
    return min_res;
}

void RayPipeline::ReservationHeap::insert_reservation(size_t node_index, float min_hit_param) {
    heap_data.push_back(NodeReservation{node_index, min_hit_param});
    std::push_heap(heap_data.begin(), heap_data.end(), NodeReservation::compare_greater);
}

bool RayPipeline::ReservationHeap::empty() {
    return heap_data.empty();
}

void RayPipeline::ReservationHeap::clear() {
    heap_data.clear();
}

void RayPipeline::RayPacket::set_ray_data(size_t index, RayData ray_data) {
    origin_x_array[index] = ray_data.origin.x;
    origin_y_array[index] = ray_data.origin.y;
    origin_z_array[index] = ray_data.origin.z;
    direction_x_array[index] = ray_data.direction.x;
    direction_y_array[index] = ray_data.direction.y;
    direction_z_array[index] = ray_data.direction.z;
    min_param_array[index] = ray_data.min_param;
    max_param_array[index] = ray_data.max_param;
    new (&reservation_heap_array[index]) ReservationHeap(std::move(ray_data.reservation_heap));
    hit_world_triangle_index_array[index] = ray_data.hit_world_triangle_index;
    extra_data_array[index] = ray_data.extra_data;
}

RayPipeline::RayData RayPipeline::RayPacket::extract_ray_data(size_t index) {
    RayData ray_data;
    ray_data.origin.x = origin_x_array[index];
    ray_data.origin.y = origin_y_array[index];
    ray_data.origin.z = origin_z_array[index];
    ray_data.direction.x = direction_x_array[index];
    ray_data.direction.y = direction_y_array[index];
    ray_data.direction.z = direction_z_array[index];
    ray_data.min_param = min_param_array[index];
    ray_data.max_param = max_param_array[index];
    ray_data.reservation_heap = std::move(reservation_heap_array[index]);
    reservation_heap_array[index].~ReservationHeap();
    ray_data.hit_world_triangle_index = hit_world_triangle_index_array[index];
    ray_data.extra_data = extra_data_array[index];
    return ray_data;
}

/*void RayPipeline::RayPacketFunnel::insert(RayPipeline& pipeline, RayData ray_data) {
    std::unique_ptr<RayPacket> new_full_packet;
    {
        auto building_view = building_per_worker.current();
        std::unique_ptr<RayPacket>& packet_ptr = *building_view;
        if (!packet_ptr) {
            packet_ptr = pipeline.create_ray_packet();
        }
        packet_ptr->set_ray_data(packet_ptr->ray_count, std::move(ray_data));
        packet_ptr->ray_count += 1;
        if (packet_ptr->ray_count == packet_ptr->capacity) {
            new_full_packet = std::move(packet_ptr);
        }
    }
    if (new_full_packet) {
        std::lock_guard<std::mutex> guard(full_packets_mutex);
        full_packets.push_back(std::move(new_full_packet));
    }
}*/

void RayPipeline::RayPacketFunnel::insert(std::unique_ptr<RayPacket> ray_packet) {
    std::lock_guard<std::mutex> guard(full_packets_mutex);
    full_packets.push_back(std::move(ray_packet));
}

bool RayPipeline::RayPacketFunnel::full_packets_empty() {
    std::lock_guard<std::mutex> guard(full_packets_mutex);
    return full_packets.empty();
}

std::unique_ptr<RayPipeline::RayPacket> RayPipeline::RayPacketFunnel::pop_full_packet() {
    std::lock_guard<std::mutex> guard(full_packets_mutex);
    std::unique_ptr<RayPacket> packet_ptr;
    if (!full_packets.empty()) {
        packet_ptr = std::move(full_packets.back());
        full_packets.pop_back();
    }
    return packet_ptr;
}

struct RayPipeline::Internal::Loader {
    World const& world;
    RayPipeline& pipeline;

    void convert_interval_to_center_extent(float vmin, float vmax, float& center, float& extent) {
        center = 0.5f * (vmin + vmax);
        extent = 0.5f * (vmax - vmin);
    }

    void convert_box_from_world(World::Box const& wbox, Box& pbox) {
        switch (pipeline.params.box_rep) {
        case RayPipelineParams::BoxRep::MinMax:
            pbox.as_minmax.min = wbox.min;
            pbox.as_minmax.max = wbox.max;
            return;
        case RayPipelineParams::BoxRep::CenterExtent:
            convert_interval_to_center_extent(wbox.min.x, wbox.max.x, pbox.as_center_extent.center.x, pbox.as_center_extent.extent.x);
            convert_interval_to_center_extent(wbox.min.y, wbox.max.y, pbox.as_center_extent.center.y, pbox.as_center_extent.extent.y);
            convert_interval_to_center_extent(wbox.min.z, wbox.max.z, pbox.as_center_extent.center.z, pbox.as_center_extent.extent.z);
            return;
        }
    }

    struct ChunkWriter {
        union PackedTriangle {
            PackedTriangleTriplePointIndexed as_triple_point_indexed;
            PackedTriangleTriplePointImmediate as_triple_point_immediate;
            PackedTriangleBoxCenteredUV as_box_centered_uv;
            PackedTriangleBoxCenteredUVExplicitN as_box_centered_uv_explicit_n;
            PackedTriangleMiddleCoordPermuted as_middle_coord_permuted;
        };

        Loader& loader;
        World::Tree const& world_tree;
        std::vector<AlignedVec3> packed_vertices;
        AlignedBuffer packed_triangles_data;
        std::vector<PackedNode> packed_nodes;
        std::vector<Box> triangle_boxes;
        std::unordered_map<size_t, int> world_to_chunk_vertex_map;
        size_t triangle_count;
        size_t triangle_size;
        ptrdiff_t triangle_index_offset;

        void set_sentinel_triangle(Box& pbox, PackedTriangle& ptri) {
            switch (loader.pipeline.params.triangle_rep) {
            case RayPipelineParams::TriangleRep::TriplePointIndexed:
                ptri.as_triple_point_indexed.vertex_indexes[0] = 0;
                ptri.as_triple_point_indexed.vertex_indexes[1] = 0;
                ptri.as_triple_point_indexed.vertex_indexes[2] = 0;
                break;
            case RayPipelineParams::TriangleRep::TriplePointImmediate:
                ptri.as_triple_point_immediate.vertices[0] = Vec3{NAN, NAN, NAN};
                ptri.as_triple_point_immediate.vertices[1] = Vec3{NAN, NAN, NAN};
                ptri.as_triple_point_immediate.vertices[2] = Vec3{NAN, NAN, NAN};
                break;
            case RayPipelineParams::TriangleRep::BoxCenteredUV:
                ptri.as_box_centered_uv.du = Vec3{NAN, NAN, NAN};
                ptri.as_box_centered_uv.dv = Vec3{NAN, NAN, NAN};
                break;
            case RayPipelineParams::TriangleRep::BoxCenteredUVExplicitN:
                ptri.as_box_centered_uv_explicit_n.du = Vec3{NAN, NAN, NAN};
                ptri.as_box_centered_uv_explicit_n.dv = Vec3{NAN, NAN, NAN};
                ptri.as_box_centered_uv_explicit_n.dn = Vec3{NAN, NAN, NAN};
                break;
            case RayPipelineParams::TriangleRep::MiddleCoordPermuted:
                ptri.as_middle_coord_permuted.mid = Vec3{NAN, NAN, NAN};
                ptri.as_middle_coord_permuted.permutation_mask = 0x15555;
                break;
            }
            switch (loader.pipeline.params.box_rep) {
            case RayPipelineParams::BoxRep::MinMax:
                pbox.as_minmax.min = Vec3{NAN, NAN, NAN};
                pbox.as_minmax.max = Vec3{NAN, NAN, NAN};
                break;
            case RayPipelineParams::BoxRep::CenterExtent:
                pbox.as_center_extent.center = Vec3{NAN, NAN, NAN};
                pbox.as_center_extent.extent = Vec3{NAN, NAN, NAN};
                break;
            }
        }

        int map_vertex_to_chunk(size_t world_index) {
            auto it = world_to_chunk_vertex_map.find(world_index);
            if (it != world_to_chunk_vertex_map.end()) {
                return it->second;
            } else {
                if (packed_vertices.size() >= 0x7fffffff) {
                    throw std::runtime_error("too many vertices in a single chunk");
                }
                int chunk_index = (int)packed_vertices.size();
                Vec3 vec = loader.world.vertices[world_index];
                packed_vertices.push_back(AlignedVec3{vec.x, vec.y, vec.z});
                world_to_chunk_vertex_map[world_index] = chunk_index;
                return chunk_index;
            }
        }

        void convert_triangle_triple_point_indexed(World::Box const& world_box, World::Triangle const& world_triangle, Box& pbox, PackedTriangle& ptri) {
            loader.convert_box_from_world(world_box, pbox);
            ptri.as_triple_point_indexed.vertex_indexes[0] = map_vertex_to_chunk(world_triangle.vertex_indexes[0]);
            ptri.as_triple_point_indexed.vertex_indexes[1] = map_vertex_to_chunk(world_triangle.vertex_indexes[1]);
            ptri.as_triple_point_indexed.vertex_indexes[2] = map_vertex_to_chunk(world_triangle.vertex_indexes[2]);
        }

        void convert_triangle_triple_point_immediate(World::Box const& world_box, World::Triangle const& world_triangle, Box& pbox, PackedTriangle& ptri) {
            loader.convert_box_from_world(world_box, pbox);
            ptri.as_triple_point_immediate.vertices[0] = loader.world.vertices[world_triangle.vertex_indexes[0]];
            ptri.as_triple_point_immediate.vertices[1] = loader.world.vertices[world_triangle.vertex_indexes[1]];
            ptri.as_triple_point_immediate.vertices[2] = loader.world.vertices[world_triangle.vertex_indexes[2]];
        }

        static float maxabs3(float a, float b, float c) {
            return fmaxf(fmaxf(fabsf(a), fabsf(b)), fabsf(c));
        }

        void convert_triangle_box_centered(World::Box const& world_box, World::Triangle const& world_triangle, Box& pbox, PackedTriangle& ptri) {
            Vec3 a = loader.world.vertices[world_triangle.vertex_indexes[0]];
            Vec3 b = loader.world.vertices[world_triangle.vertex_indexes[1]];
            Vec3 c = loader.world.vertices[world_triangle.vertex_indexes[2]];
            Vec3 center = (1.0f / 3.0f) * (a + b + c);
            Vec3 rel_a = a - center;
            Vec3 rel_b = b - center;
            Vec3 rel_c = c - center;
            Vec3 extent = Vec3{
                maxabs3(rel_a.x, rel_b.x, rel_c.x),
                maxabs3(rel_a.y, rel_b.y, rel_c.y),
                maxabs3(rel_a.z, rel_b.z, rel_c.z),
            };
            Vec3 u = b - a;
            Vec3 v = c - a;
            Vec3 n = cross(u, v);
            float nsqr = dotsqr(n);
            Vec3 dn = (3.0f / nsqr) * n;
            Vec3 du = cross(v, dn);
            Vec3 dv = cross(dn, u);
            switch (loader.pipeline.params.box_rep) {
            case RayPipelineParams::BoxRep::MinMax:
                pbox.as_minmax.min = center - extent;
                pbox.as_minmax.max = center + extent;
                break;
            case RayPipelineParams::BoxRep::CenterExtent:
                pbox.as_center_extent.center = center;
                pbox.as_center_extent.extent = extent;
                break;
            }
            switch (loader.pipeline.params.triangle_rep) {
            case RayPipelineParams::TriangleRep::BoxCenteredUV:
                ptri.as_box_centered_uv.du = du;
                ptri.as_box_centered_uv.dv = dv;
                break;
            case RayPipelineParams::TriangleRep::BoxCenteredUVExplicitN:
                ptri.as_box_centered_uv_explicit_n.du = du;
                ptri.as_box_centered_uv_explicit_n.dv = dv;
                ptri.as_box_centered_uv_explicit_n.dn = dn;
                break;
            default:
                throw std::runtime_error("invalid triangle rep");
            }
        }

        static void index_sort3(float const(&values)[3], int(&indexes)[3], int(&places)[3]) {
            indexes[0] = 0;
            indexes[1] = 1;
            indexes[2] = 2;
            /* an unrolled version of the insertion sort */
            if (values[indexes[0]] > values[indexes[1]]) {
                std::swap(indexes[0], indexes[1]);
            }
            if (values[indexes[1]] > values[indexes[2]]) {
                std::swap(indexes[1], indexes[2]);
            }
            if (values[indexes[0]] > values[indexes[1]]) {
                std::swap(indexes[0], indexes[1]);
            }
            places[indexes[0]] = 0;
            places[indexes[1]] = 1;
            places[indexes[2]] = 2;
        }

        void convert_triangle_middle_point_permuted(World::Box const& world_box, World::Triangle const& world_triangle, Box& pbox, PackedTriangle& ptri) {
            Vec3 a = loader.world.vertices[world_triangle.vertex_indexes[0]];
            Vec3 b = loader.world.vertices[world_triangle.vertex_indexes[1]];
            Vec3 c = loader.world.vertices[world_triangle.vertex_indexes[2]];
            float xvalues[3] = {a.x, b.x, c.x};
            float yvalues[3] = {a.y, b.y, c.y};
            float zvalues[3] = {a.z, b.z, c.z};
            int xorder[3], yorder[3], zorder[3];
            int xplace[3], yplace[3], zplace[3];
            index_sort3(xvalues, xorder, xplace);
            index_sort3(yvalues, yorder, yplace);
            index_sort3(zvalues, zorder, zplace);
            Vec3 min = Vec3{xvalues[xorder[0]], yvalues[yorder[0]], zvalues[zorder[0]]};
            Vec3 mid = Vec3{xvalues[xorder[1]], yvalues[yorder[1]], zvalues[zorder[1]]};
            Vec3 max = Vec3{xvalues[xorder[2]], yvalues[yorder[2]], zvalues[zorder[2]]};
            uint32_t permutation_mask =
                (uint32_t)(xplace[0]) |
                ((uint32_t)(yplace[0]) << 2) |
                ((uint32_t)(zplace[0]) << 4) |
                ((uint32_t)(xplace[1]) << 6) |
                ((uint32_t)(yplace[1]) << 8) |
                ((uint32_t)(zplace[1]) << 10) |
                ((uint32_t)(xplace[2]) << 12) |
                ((uint32_t)(yplace[2]) << 14) |
                ((uint32_t)(zplace[2]) << 16);
            switch (loader.pipeline.params.box_rep) {
            case RayPipelineParams::BoxRep::MinMax:
                pbox.as_minmax.min = min;
                pbox.as_minmax.max = max;
                break;
            case RayPipelineParams::BoxRep::CenterExtent:
                pbox.as_center_extent.center = 0.5f * (min + max);
                pbox.as_center_extent.extent = 0.5f * (max - min);
                break;
            }
            ptri.as_middle_coord_permuted.mid = mid;
            ptri.as_middle_coord_permuted.permutation_mask = permutation_mask;
        }

        void convert_triangle(World::Box const& world_box, World::Triangle const& world_triangle, Box& pbox, PackedTriangle& ptri) {
            switch (loader.pipeline.params.triangle_rep) {
            case RayPipelineParams::TriangleRep::TriplePointIndexed:
                return convert_triangle_triple_point_indexed(world_box, world_triangle, pbox, ptri);
            case RayPipelineParams::TriangleRep::TriplePointImmediate:
                return convert_triangle_triple_point_immediate(world_box, world_triangle, pbox, ptri);
            case RayPipelineParams::TriangleRep::BoxCenteredUV:
            case RayPipelineParams::TriangleRep::BoxCenteredUVExplicitN:
                return convert_triangle_box_centered(world_box, world_triangle, pbox, ptri);
            case RayPipelineParams::TriangleRep::MiddleCoordPermuted:
                return convert_triangle_middle_point_permuted(world_box, world_triangle, pbox, ptri);
            }
        }

        void write_chunk_triangles(size_t root_node_level, size_t root_node_index) {
            size_t index_begin = root_node_index;
            size_t index_end = root_node_index + 1;
            for (size_t level = root_node_level; level > 0; --level) {
                index_begin = world_tree.node_children_indexes[level - 1][index_begin];
                index_end = world_tree.node_children_indexes[level - 1][index_end];
            }
            triangle_count = index_end - index_begin;
            if (triangle_count > 0x7ffffffe) {
                throw std::runtime_error("too many triangles in a chunk");
            }
            triangle_size = loader.pipeline.params.sizeof_packed_triangle();
            packed_triangles_data = AlignedBuffer(triangle_size * (triangle_count + 1));
            triangle_boxes.resize(triangle_count + 1);
            set_sentinel_triangle(triangle_boxes.front(), *(PackedTriangle*)packed_triangles_data.data());
            for (size_t i = 0; i < triangle_count; ++i) {
                convert_triangle(
                    world_tree.node_boxes[0][index_begin + i],
                    world_tree.triangles[index_begin + i],
                    triangle_boxes[i + 1],
                    *(PackedTriangle*)(packed_triangles_data.data() + triangle_size * (i + 1)));
            }
            triangle_index_offset = (ptrdiff_t)index_begin - 1;
        }

        void write_node(size_t node_level, size_t node_index) {
            size_t chunk_node_index = packed_nodes.size();
            if (node_level == 0) {
                PackedNode& pnode = packed_nodes.emplace_back();
                int chunk_triangle_index = (int)(node_index - triangle_index_offset);
                pnode.box = triangle_boxes[chunk_triangle_index];
                pnode.subtree_size = 1;
                pnode.triangle_index = chunk_triangle_index;
            } else {
                {
                    PackedNode& pnode = packed_nodes.emplace_back();
                    loader.convert_box_from_world(world_tree.node_boxes[node_level][node_index], pnode.box);
                }
                size_t children_world_begin = world_tree.node_children_indexes[node_level - 1][node_index];
                size_t children_world_end = world_tree.node_children_indexes[node_level - 1][node_index + 1];
                for (size_t child_index = children_world_begin; child_index < children_world_end; ++child_index) {
                    write_node(node_level - 1, child_index);
                }
                size_t next_node_count = packed_nodes.size();
                size_t subtree_size = next_node_count - chunk_node_index;
                if (subtree_size > 0x7ffffffe) {
                    throw std::runtime_error("too many nodes in a chunk");
                }
                {
                    PackedNode& pnode = packed_nodes[chunk_node_index];
                    pnode.subtree_size = (int)subtree_size;
                    pnode.triangle_index = 0;
                }
            }
        }

        void write_node_root(size_t node_level, size_t node_index) {
            if (node_level == 0) {
                write_node(node_level, node_index);
            } else {
                size_t children_world_begin = world_tree.node_children_indexes[node_level - 1][node_index];
                size_t children_world_end = world_tree.node_children_indexes[node_level - 1][node_index + 1];
                for (size_t child_index = children_world_begin; child_index < children_world_end; ++child_index) {
                    write_node(node_level - 1, child_index);
                }
            }
        }

        void write_sentinel_node() {
            PackedNode& pnode = packed_nodes.emplace_back();
            pnode.box = triangle_boxes[0];
            pnode.subtree_size = 0;
            pnode.triangle_index = 0;
        }

        std::unique_ptr<Chunk> produce_chunk() {
            std::unique_ptr<Chunk> chunk_ptr = std::make_unique<Chunk>();
            chunk_ptr->vertex_count = (int)packed_vertices.size();
            chunk_ptr->triangle_count = (int)triangle_count;
            chunk_ptr->sentinel_node_index = (int)(packed_nodes.size() - 1);
            chunk_ptr->triangle_index_offset = triangle_index_offset;
            
            AlignedBufferLayout buffer_layout;
            buffer_layout.add_item_vector(packed_vertices, chunk_ptr->vertices);
            buffer_layout.add_item(packed_triangles_data.data(), (void**)&chunk_ptr->triangles, triangle_count + 1, triangle_size, loader.pipeline.params.alignof_packed_triangle());
            buffer_layout.add_item_vector(packed_nodes, chunk_ptr->nodes);
            
            chunk_ptr->buffer = AlignedBuffer(buffer_layout.total_size());
            
            buffer_layout.transfer_data(chunk_ptr->buffer.data());
            
            return chunk_ptr;
        }
    };

    std::unique_ptr<Chunk> create_chunk(World::Tree const& world_tree, size_t node_level, size_t node_index) {
        ChunkWriter writer{*this, world_tree};
        writer.packed_vertices.push_back(AlignedVec3{NAN, NAN, NAN});
        writer.write_chunk_triangles(node_level, node_index);
        writer.write_node_root(node_level, node_index);
        writer.write_sentinel_node();
        return writer.produce_chunk();
    }

    struct TempNode {
        size_t level, internal_index, child_start_index;
        Box box;
        std::vector<TempNode> children;
        std::unique_ptr<Chunk> chunk_ptr;
    };
    
    TempNode create_temp_node(World::Tree const& world_tree, size_t node_level, size_t node_index) {
        TempNode tnode;
        convert_box_from_world(world_tree.node_boxes[node_level][node_index], tnode.box);
        if (node_level <= pipeline.params.max_chunk_height) {
            tnode.level = 0;
            tnode.chunk_ptr = create_chunk(world_tree, node_level, node_index);
        } else {
            tnode.level = node_level - pipeline.params.max_chunk_height;
            size_t children_start = world_tree.node_children_indexes[node_level - 1][node_index];
            size_t children_end = world_tree.node_children_indexes[node_level - 1][node_index + 1];
            size_t children_count = children_end - children_start;
            tnode.children.resize(children_count);
            for (size_t i = 0; i < children_count; ++i) {
                tnode.children[i] = create_temp_node(world_tree, node_level - 1, children_start + i);
            }
        }
        return tnode;
    }
    
    void enumerate_temp_node_internal(TempNode& tnode, std::vector<size_t>& level_counters) {
        tnode.internal_index = level_counters[tnode.level];
        if (tnode.level >= 1) {
            tnode.child_start_index = level_counters[tnode.level - 1];
        } else {
            tnode.child_start_index = 0;
        }
        level_counters[tnode.level] += 1;
        for (TempNode& child : tnode.children) {
            enumerate_temp_node_internal(child, level_counters);
        }
    }
    
    void pack_temp_node(Tree& ptree, TempNode& tnode, std::vector<size_t> const& level_offsets) {
        size_t global_index = tnode.internal_index + level_offsets[tnode.level];
        ptree.node_boxes[global_index] = tnode.box;
        if (tnode.level == 0) {
            ptree.chunks[global_index] = std::move(tnode.chunk_ptr);
        } else {
            size_t leaf_count = level_offsets[1];
            size_t shifted_index = global_index - leaf_count;
            size_t child_global_index = tnode.child_start_index + level_offsets[tnode.level - 1];
            ptree.node_child_indexes[shifted_index] = child_global_index;
            for (TempNode& child : tnode.children) {
                pack_temp_node(ptree, child, level_offsets);
            }
        }
    }
    
    std::unique_ptr<Tree> convert_temp_nodes_to_packed_tree(TempNode& root_tnode, std::vector<size_t>& level_counters) {
        std::unique_ptr<Tree> ptree_ptr = std::make_unique<Tree>();
        size_t total_chunk_count = level_counters[0];
        size_t total_node_count = 0;
        for (size_t& cref : level_counters) {
            size_t level_pop = cref;
            cref = total_node_count;
            total_node_count += level_pop;
        }
        size_t total_branch_count = total_node_count - total_chunk_count;
        ptree_ptr->chunks.resize(total_chunk_count);
        ptree_ptr->node_boxes.resize(total_node_count);
        ptree_ptr->node_child_indexes.resize(total_branch_count + 1);
        pack_temp_node(*ptree_ptr, root_tnode, level_counters);
        ptree_ptr->node_child_indexes[total_branch_count] = total_node_count - 1;
        return ptree_ptr;
    }

    void initialize_processor_objects();

    void load_pipeline() {
        size_t zone_count = world.zone_trees.size();
        pipeline.zone_trees.resize(zone_count);
        for (size_t i = 0; i < zone_count; ++i) {
            World::Tree const& world_tree = world.zone_trees[i];
            TempNode root_tnode = create_temp_node(world_tree, world_tree.node_boxes.size() - 1, 0);
            std::vector<size_t> level_counters(root_tnode.level + 1, 0);
            enumerate_temp_node_internal(root_tnode, level_counters);
            pipeline.zone_trees[i] = convert_temp_nodes_to_packed_tree(root_tnode, level_counters);
            pipeline.zone_trees[i]->index = i;
        }
        pipeline.processor_control = std::make_shared<ProcessorControl>();
        initialize_processor_objects();
    }
};

RayPipeline::RayPipeline(World const& world, RayPipelineParams const& params) {
    this->params = params;
    Internal::Loader loader{world, *this};
    loader.load_pipeline();
}

RayPipeline::~RayPipeline() {
    processor_control->set_dead();
}

RayPipelineParams const& RayPipeline::get_params() {
    return params;
}

std::unique_ptr<RayPipeline::RayPacket> RayPipeline::create_ray_packet(size_t zone_id) {
    std::unique_ptr<RayPacket> packet_ptr;
    {
        std::lock_guard<std::mutex> guard(ray_packets_spare_mutex);
        if (!ray_packets_spare.empty()) {
            packet_ptr = std::move(ray_packets_spare.back());
            ray_packets_spare.pop_back();
        }
    }
    if (!packet_ptr) {
        int capacity = params.ray_buffer_size;
        packet_ptr = std::make_unique<RayPacket>();
        packet_ptr->capacity = capacity;
        packet_ptr->ray_count = 0;

        AlignedBufferLayout buffer_layout;
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->origin_x_array, capacity, sizeof(float), 32);
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->origin_y_array, capacity, sizeof(float), 32);
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->origin_z_array, capacity, sizeof(float), 32);
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->direction_x_array, capacity, sizeof(float), 32);
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->direction_y_array, capacity, sizeof(float), 32);
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->direction_z_array, capacity, sizeof(float), 32);
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->min_param_array, capacity, sizeof(float), 32);
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->max_param_array, capacity, sizeof(float), 32);
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->reservation_heap_array, capacity, sizeof(ReservationHeap), alignof(ReservationHeap));
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->hit_world_triangle_index_array, capacity, sizeof(size_t), alignof(size_t));
        buffer_layout.add_item(nullptr, (void**)&packet_ptr->extra_data_array, capacity, sizeof(void*), alignof(void*));

        packet_ptr->buffer = AlignedBuffer(buffer_layout.total_size());

        buffer_layout.transfer_data(packet_ptr->buffer.data());

    }
    packet_ptr->id = next_packet_id.fetch_add(1, std::memory_order_relaxed);
    packet_ptr->zone_id = zone_id;
    total_packet_count.fetch_add(1, std::memory_order_relaxed);
    return packet_ptr;
}

void RayPipeline::collect_ray_packet_spare(std::unique_ptr<RayPacket> packet_ptr) {
    total_packet_count.fetch_sub(1, std::memory_order_relaxed);
    packet_ptr->ray_count = 0;
    {
        std::lock_guard<std::mutex> guard(ray_packets_spare_mutex);
        ray_packets_spare.push_back(std::move(packet_ptr));
    }
}

/*std::unique_ptr<RayPipeline::RayBuffer2> RayPipeline::create_ray_buffer_2() {
    RayBuffer2 buffer;
    buffer.reserve(params.ray_buffer_size);
    return std::make_unique<RayBuffer2>(std::move(buffer));
}*/

struct RayPipeline::Internal::Utils {
    static void get_coord_param_interval_minmax(float ray_origin, float ray_dir_inverse, float box_min, float box_max, float& minp, float& maxp) {
        float p1 = (box_min - ray_origin) * ray_dir_inverse;
        float p2 = (box_max - ray_origin) * ray_dir_inverse;
        if (p1 <= p2) {
            minp = p1;
            maxp = p2;
        } else if (p1 > p2) {
            minp = p2;
            maxp = p1;
        }
    }

    static void get_coord_param_interval_center_extent(float ray_origin, float ray_dir_inverse, float box_center, float box_extent, float& minp, float& maxp) {
        float pc = (box_center - ray_origin) * ray_dir_inverse;
        float pd = fabsf(box_extent * ray_dir_inverse);
        if (pd >= 0) {
            minp = pc - pd;
            maxp = pc + pd;
        }
    }

    static void clip_param_interval(RayPipelineParams const& params, Vec3 origin, Vec3 dir_inverse, Box const& box, float& minp, float& maxp) {
        float minpx, maxpx, minpy, maxpy, minpz, maxpz;
        switch (params.box_rep) {
        case RayPipelineParams::BoxRep::MinMax:
            get_coord_param_interval_minmax(origin.x, dir_inverse.x, box.as_minmax.min.x, box.as_minmax.max.x, minpx, maxpx);
            get_coord_param_interval_minmax(origin.y, dir_inverse.y, box.as_minmax.min.y, box.as_minmax.max.y, minpy, maxpy);
            get_coord_param_interval_minmax(origin.z, dir_inverse.z, box.as_minmax.min.z, box.as_minmax.max.z, minpz, maxpz);
            break;
        case RayPipelineParams::BoxRep::CenterExtent:
            get_coord_param_interval_center_extent(origin.x, dir_inverse.x, box.as_center_extent.center.x, box.as_center_extent.extent.x, minpx, maxpx);
            get_coord_param_interval_center_extent(origin.y, dir_inverse.y, box.as_center_extent.center.y, box.as_center_extent.extent.y, minpy, maxpy);
            get_coord_param_interval_center_extent(origin.z, dir_inverse.z, box.as_center_extent.center.z, box.as_center_extent.extent.z, minpz, maxpz);
            break;
        }
        /* NaNs poison the final minp/maxp */
        if (!(minpx <= minp)) minp = minpx;
        if (!(minpy <= minp)) minp = minpy;
        if (!(minpz <= minp)) minp = minpz;
        if (!(maxpx >= maxp)) maxp = maxpx;
        if (!(maxpy >= maxp)) maxp = maxpy;
        if (!(maxpz >= maxp)) maxp = maxpz;
    }

    static Vec3 get_box_center(RayPipelineParams const& params, Box const& box) {
        switch (params.box_rep) {
        case RayPipelineParams::BoxRep::MinMax:
            return 0.5f * (box.as_minmax.min + box.as_minmax.max);
        case RayPipelineParams::BoxRep::CenterExtent:
            return box.as_center_extent.center;
        default:
            throw std::runtime_error("invalid box representation value");
        }
    }

    static void get_box_minmax(RayPipelineParams const& params, Box const& box, Vec3& min, Vec3& max) {
        switch (params.box_rep) {
        case RayPipelineParams::BoxRep::MinMax:
            min = box.as_minmax.min;
            max = box.as_minmax.max;
            break;
        case RayPipelineParams::BoxRep::CenterExtent:
            min = box.as_center_extent.center - box.as_center_extent.extent;
            max = box.as_center_extent.center + box.as_center_extent.extent;
            break;
        default:
            throw std::runtime_error("invalid box representation value");
        }
    }
};

struct RayPipeline::Internal::ChunkIntersectorImmediate {
    RayPipelineParams const& params;
    Chunk const& chunk;
    Vec3 ray_origin;
    Vec3 ray_direction;
    Vec3 ray_dir_inverse;
    Ray::SpaceShear ray_space_shear;
    float ray_min_param;
    float ray_max_param;
    std::vector<int> pnode_stack;
    int hit_triangle_index = 0;
    Vec3 hit_triangle_pos;
    
    void intersect_triangle_triple_point_common(PackedNode const& pnode, Vec3 a, Vec3 b, Vec3 c, int triangle_index) {
        a = a - ray_origin;
        b = b - ray_origin;
        c = c - ray_origin;
        float ax = a[ray_space_shear.kx] - ray_space_shear.sx * a[ray_space_shear.kz];
        float ay = a[ray_space_shear.ky] - ray_space_shear.sy * a[ray_space_shear.kz];
        float bx = b[ray_space_shear.kx] - ray_space_shear.sx * b[ray_space_shear.kz];
        float by = b[ray_space_shear.ky] - ray_space_shear.sy * b[ray_space_shear.kz];
        float cx = c[ray_space_shear.kx] - ray_space_shear.sx * c[ray_space_shear.kz];
        float cy = c[ray_space_shear.ky] - ray_space_shear.sy * c[ray_space_shear.kz];
        float u = cx * by - cy * bx;
        float v = ax * cy - ay * cx;
        float w = bx * ay - by * ax;
        if (u < 0 || v < 0 || w < 0) {
            return;
        }
        float det = u + v + w;
        if (det == 0) {
            return;
        }
        float az = ray_space_shear.sz * a[ray_space_shear.kz];
        float bz = ray_space_shear.sz * b[ray_space_shear.kz];
        float cz = ray_space_shear.sz * c[ray_space_shear.kz];
        float t = u * az + v * bz + w * cz;
        float scaled_min_param = ray_min_param * det;
        float scaled_max_param = ray_max_param * det;
        if (scaled_min_param <= t && t <= scaled_max_param) {
            float param = t / det;
            ray_max_param = param;
            hit_triangle_index = triangle_index;
            hit_triangle_pos = Vec3{(float)(u / det), (float)(v / det), (float)(w / det)};
        }
    }
    
    void intersect_triangle_box_centered_common(PackedNode const& pnode, Vec3 du, Vec3 dv, Vec3 dn, int triangle_index) {
        float direction_n = dot(ray_direction, dn);
        if (direction_n <= 0) {
            Vec3 origin_relative = ray_origin - Utils::get_box_center(params, pnode.box);
            float origin_n = dot(origin_relative, dn);
            float param = -origin_n / direction_n;
            if (ray_min_param <= param && param <= ray_max_param) {
                Vec3 rel_hit = origin_relative + param * ray_direction;
                float hit_u = dot(rel_hit, du);
                float hit_v = dot(rel_hit, dv);
                if (hit_u >= -1 && hit_v >= -1 && hit_u + hit_v <= 1) {
                    ray_max_param = param;
                    hit_triangle_index = triangle_index;
                    float u = 0.5f * (hit_u + 1.0f);
                    float v = 0.5f * (hit_v + 1.0f);
                    float w = 1 - u - v;
                    hit_triangle_pos = Vec3{u, v, w};
                }
            }
        }
    }
    
    void intersect_triangle(PackedNode const& pnode, int index) {
        switch (params.triangle_rep) {
        case RayPipelineParams::TriangleRep::TriplePointIndexed:
        {
            auto ptri = ((PackedTriangleTriplePointIndexed*)chunk.triangles)[index];
            Vec3 a = chunk.vertices[ptri.vertex_indexes[0]];
            Vec3 b = chunk.vertices[ptri.vertex_indexes[1]];
            Vec3 c = chunk.vertices[ptri.vertex_indexes[2]];
            return intersect_triangle_triple_point_common(pnode, a, b, c, index);
        }
        case RayPipelineParams::TriangleRep::TriplePointImmediate:
        {
            auto ptri = ((PackedTriangleTriplePointImmediate*)chunk.triangles)[index];
            Vec3 a = ptri.vertices[0];
            Vec3 b = ptri.vertices[1];
            Vec3 c = ptri.vertices[2];
            return intersect_triangle_triple_point_common(pnode, a, b, c, index);
        }
        case RayPipelineParams::TriangleRep::BoxCenteredUV:
        {
            auto ptri = ((PackedTriangleBoxCenteredUV*)chunk.triangles)[index];
            Vec3 dn = cross(ptri.du, ptri.dv);
            return intersect_triangle_box_centered_common(pnode, ptri.du, ptri.dv, dn, index);
        }
        case RayPipelineParams::TriangleRep::BoxCenteredUVExplicitN:
        {
            auto ptri = ((PackedTriangleBoxCenteredUVExplicitN*)chunk.triangles)[index];
            return intersect_triangle_box_centered_common(pnode, ptri.du, ptri.dv, ptri.dn, index);
        }
        case RayPipelineParams::TriangleRep::MiddleCoordPermuted:
        {
            auto ptri = ((PackedTriangleMiddleCoordPermuted*)chunk.triangles)[index];
            Vec3 min, max;
            Utils::get_box_minmax(params, pnode.box, min, max);
            Vec3 mid = ptri.mid;
            float xs[3] = {min.x, mid.x, max.x};
            float ys[3] = {min.y, mid.y, max.y};
            float zs[3] = {min.z, mid.z, max.z};
            uint32_t mask = ptri.permutation_mask;
            Vec3 a{xs[mask & 3], ys[(mask >> 2) & 3], zs[(mask >> 4) & 3]};
            Vec3 b{xs[(mask >> 6) & 3], ys[(mask >> 8) & 3], zs[(mask >> 10) & 3]};
            Vec3 c{xs[(mask >> 12) & 3], ys[(mask >> 14) & 3], zs[(mask >> 16) & 3]};
            return intersect_triangle_triple_point_common(pnode, a, b, c, index);
        }
        }
    }

    void intersect_all() {
        int pnode_index = 0;
        while (pnode_index < chunk.sentinel_node_index) {
            PackedNode const& pnode = chunk.nodes[pnode_index];
            float minp = ray_min_param, maxp = ray_max_param;
            Utils::clip_param_interval(params, ray_origin, ray_dir_inverse, pnode.box, minp, maxp);
            if (minp <= maxp) {
                int triangle_index = pnode.triangle_index;
                if (triangle_index > 0) {
                    intersect_triangle(pnode, triangle_index);
                }
                pnode_index += 1;
            } else {
                pnode_index += pnode.subtree_size;
            }
        }
    }
};

void RayPipeline::trace_immediately(Ray& ray) {
    if (ray._zone_index >= zone_trees.size()) {
        return;
    }
    Tree const& ptree = *zone_trees[ray._zone_index];

    Vec3 ray_origin = ray._origin;
    Vec3 ray_direction = ray._direction;
    Vec3 ray_dir_inverse = elementwise_inverse(ray_direction);
    Ray::SpaceShear ray_space_shear = ray._space_shear;

    size_t total_chunk_count = ptree.chunks.size();

    ray._reservation_min_heap.clear();

    auto visit_node = [&](size_t node_index) {
        float minp = ray._min_param, maxp = ray._max_param;
        Internal::Utils::clip_param_interval(params, ray_origin, ray_dir_inverse, ptree.node_boxes[node_index], minp, maxp);
        if (minp <= maxp) {
            ray.insert_reservation(node_index, minp);
        }
    };

    visit_node(ptree.node_boxes.size() - 1);

    while (!ray._reservation_min_heap.empty()) {
        Ray::NodeReservation min_res = ray.pop_min_reservation();
        if (min_res.min_hit_param > ray._max_param) {
            ray._reservation_min_heap.clear();
            break;
        }

        size_t node_index = min_res.node_index;

        if (node_index < total_chunk_count) {
            Internal::ChunkIntersectorImmediate chunk_intersector{
                params,
                *ptree.chunks[node_index],
                ray_origin,
                ray_direction,
                ray_dir_inverse,
                ray_space_shear,
                ray._min_param,
                ray._max_param,
            };
            chunk_intersector.intersect_all();
            if (chunk_intersector.hit_triangle_index > 0) {
                ray._min_param = chunk_intersector.ray_min_param;
                ray._max_param = chunk_intersector.ray_max_param;
                ray._hit_triangle_index = chunk_intersector.chunk.triangle_index_offset + (size_t)chunk_intersector.hit_triangle_index;
                ray._hit_triangle_pos = chunk_intersector.hit_triangle_pos;
            }
        } else {
            size_t children_begin = ptree.node_child_indexes[node_index - total_chunk_count];
            size_t children_end = ptree.node_child_indexes[node_index - total_chunk_count + 1];
            for (size_t i = children_begin; i < children_end; ++i) {
                visit_node(i);
            }
        }
    }
}

struct RayPipeline::Internal::RayScheduler {
    RayPipeline& pipeline;

    void dispatch_ray(std::unique_ptr<Ray> ray_ptr);
};

struct RayPipeline::Internal::ChunkIntersectorSimd {
    struct YmmBox {
        YmmFloat v[6];
    };

    struct YmmVec3 {
        YmmFloat x, y, z;

        friend YmmVec3 operator+(YmmVec3 a, YmmVec3 b) {
            return YmmVec3{a.x + b.x, a.y + b.y, a.z + b.z};
        }

        friend YmmVec3 operator-(YmmVec3 a, YmmVec3 b) {
            return YmmVec3{a.x - b.x, a.y - b.y, a.z - b.z};
        }

        friend YmmVec3 operator*(YmmFloat a, YmmVec3 b) {
            return YmmVec3{a * b.x, a * b.y, a * b.z};
        }

        friend YmmVec3 cross(YmmVec3 a, YmmVec3 b) {
            return YmmVec3{
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x,
            };
        }

        friend YmmFloat dot(YmmVec3 a, YmmVec3 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        friend YmmFloat det3(YmmVec3 a, YmmVec3 b, YmmVec3 c) {
            return dot(cross(a, b), c);
        }
    };

    struct YmmSpaceShear {
        YmmInt permutation;
        YmmFloat sx, sy, sz;

        static YmmSpaceShear make_for(YmmVec3 dir) {
            YmmFloatData dx_data = dir.x.data();
            YmmFloatData dy_data = dir.y.data();
            YmmFloatData dz_data = dir.z.data();
            YmmIntData permutation_data;
            YmmFloatData sx_data, sy_data, sz_data;
            for (int i = 0; i < 8; ++i) {
                float d[3] = {dx_data.x[i], dy_data.x[i], dz_data.x[i]};
                int max_dim = 0;
                if (fabsf(d[1]) > fabsf(d[0])) {
                    max_dim = 1;
                }
                if (fabsf(d[2]) > fabsf(d[max_dim])) {
                    max_dim = 2;
                }
                int kz = max_dim;
                int kx = (kz + 1) % 3;
                int ky = (kx + 1) % 3;
                if (d[kz] < 0) {
                    std::swap(kx, ky);
                }
                permutation_data.x[i] = (int32_t)(((uint32_t)kx << 30) | ((uint32_t)ky << 28) | ((uint32_t)kz << 26));
                sx_data.x[i] = d[kx] / d[kz];
                sy_data.x[i] = d[ky] / d[kz];
                sz_data.x[i] = 1.0f / d[kz];
            }
            YmmSpaceShear ss;
            ss.permutation = YmmInt(permutation_data);
            ss.sx = YmmFloat(sx_data);
            ss.sy = YmmFloat(sy_data);
            ss.sz = YmmFloat(sz_data);
            return ss;
        }

        YmmVec3 apply_permutation(YmmVec3 v) {
            YmmFloat nx = _mm256_blendv_ps(_mm256_blendv_ps(v.x, v.y, YmmFloat(_mm256_slli_epi32(permutation, 1))), v.z, YmmFloat(permutation));
            YmmFloat ny = _mm256_blendv_ps(_mm256_blendv_ps(v.x, v.y, YmmFloat(_mm256_slli_epi32(permutation, 3))), v.z, YmmFloat(_mm256_slli_epi32(permutation, 2)));
            YmmFloat nz = _mm256_blendv_ps(_mm256_blendv_ps(v.x, v.y, YmmFloat(_mm256_slli_epi32(permutation, 5))), v.z, YmmFloat(_mm256_slli_epi32(permutation, 4)));
            return YmmVec3{nx, ny, nz};
        }
    };

    RayPipeline& pipeline;
    Chunk& chunk;
    YmmVec3 origin;
    YmmVec3 direction;
    YmmVec3 dir_inverse;
    YmmFloat min_param;
    YmmFloat max_param;
    YmmInt hit_triangle_index;
    YmmSpaceShear ray_space_shear;

    static void get_coord_param_interval_minmax(YmmFloat ray_origin, YmmFloat ray_dir_inverse, YmmFloat box_min, YmmFloat box_max, YmmFloat& minp, YmmFloat& maxp) {
        YmmFloat p1 = (box_min - ray_origin) * ray_dir_inverse;
        YmmFloat p2 = (box_max - ray_origin) * ray_dir_inverse;
        minp = YmmFloat::min(p1, p2);
        maxp = YmmFloat::max(p1, p2);
    }
    
    static void get_coord_param_interval_center_extent(YmmFloat ray_origin, YmmFloat ray_dir_inverse, YmmFloat box_center, YmmFloat box_extent, YmmFloat& minp, YmmFloat& maxp) {
        YmmFloat pc = (box_center - ray_origin) * ray_dir_inverse;
        YmmFloat pd = (box_extent * ray_dir_inverse).abs();
        minp = pc - pd;
        maxp = pc + pd;
    }
    
    void clip_param_interval(YmmBox box, YmmFloat& minp, YmmFloat& maxp) {
        YmmFloat minpx, maxpx, minpy, maxpy, minpz, maxpz;
        switch (pipeline.params.box_rep) {
        case RayPipelineParams::BoxRep::MinMax:
            get_coord_param_interval_minmax(origin.x, dir_inverse.x, box.v[0], box.v[3], minpx, maxpx);
            get_coord_param_interval_minmax(origin.y, dir_inverse.y, box.v[1], box.v[4], minpy, maxpy);
            get_coord_param_interval_minmax(origin.z, dir_inverse.z, box.v[2], box.v[5], minpz, maxpz);
            break;
        case RayPipelineParams::BoxRep::CenterExtent:
            get_coord_param_interval_center_extent(origin.x, dir_inverse.x, box.v[0], box.v[3], minpx, maxpx);
            get_coord_param_interval_center_extent(origin.y, dir_inverse.y, box.v[1], box.v[4], minpy, maxpy);
            get_coord_param_interval_center_extent(origin.z, dir_inverse.z, box.v[2], box.v[5], minpz, maxpz);
            break;
        }
        minp = YmmFloat::max(minp, minpx);
        minp = YmmFloat::max(minp, minpy);
        minp = YmmFloat::max(minp, minpz);
        maxp = YmmFloat::min(maxp, maxpx);
        maxp = YmmFloat::min(maxp, maxpy);
        maxp = YmmFloat::min(maxp, maxpz);
    }
    
    void process_triangle_triple_point_common(YmmInt is_triangle_int, YmmInt triangle_index, YmmVec3 wa, YmmVec3 wb, YmmVec3 wc) {
        YmmFloat is_triangle = YmmFloat(is_triangle_int);
        YmmVec3 a = wa - origin;
        YmmVec3 b = wb - origin;
        YmmVec3 c = wc - origin;
        YmmVec3 ap = ray_space_shear.apply_permutation(a);
        YmmVec3 bp = ray_space_shear.apply_permutation(b);
        YmmVec3 cp = ray_space_shear.apply_permutation(c);
        YmmFloat apx = ap.x - ray_space_shear.sx * ap.z;
        YmmFloat apy = ap.y - ray_space_shear.sy * ap.z;
        YmmFloat bpx = bp.x - ray_space_shear.sx * bp.z;
        YmmFloat bpy = bp.y - ray_space_shear.sy * bp.z;
        YmmFloat cpx = cp.x - ray_space_shear.sx * cp.z;
        YmmFloat cpy = cp.y - ray_space_shear.sy * cp.z;
        YmmFloat u = cpx * bpy - cpy * bpx;
        YmmFloat v = apx * cpy - apy * cpx;
        YmmFloat w = bpx * apy - bpy * apx;
        //YmmFloat u2 = -det3(direction, a, b);
        //YmmFloat v2 = -det3(direction, b, c);
        //YmmFloat w2 = -det3(direction, c, a);
        YmmFloat is_interior = is_triangle & (u >= YmmFloat::set1(0)) & (v >= YmmFloat::set1(0)) & (w >= YmmFloat::set1(0));
        YmmFloat det = u + v + w;
        //YmmFloat det2 = u2 + v2 + w2;
        YmmFloat is_valid_hit = is_interior & (det != YmmFloat::set1(0));
        if (is_valid_hit.any()) {
            YmmFloat apz = ray_space_shear.sz * ap.z;
            YmmFloat bpz = ray_space_shear.sz * bp.z;
            YmmFloat cpz = ray_space_shear.sz * cp.z;
            YmmFloat t = u * apz + v * bpz + w * cpz;
            //YmmFloat t2 = -det3(a, b, c);
            YmmFloat scaled_min_param = min_param * det;
            YmmFloat scaled_max_param = max_param * det;
            YmmFloat is_valid_param = is_valid_hit & (scaled_min_param <= t) & (t <= scaled_max_param);
            if (is_valid_param.any()) {
                YmmFloat param = t * det.recip();
                //YmmFloat param2 = t2 * det2.recip();
                max_param = YmmFloat::blend(max_param, param, is_valid_param);
                hit_triangle_index = YmmInt::blend(hit_triangle_index, triangle_index, YmmInt(is_valid_param));
            }
        }
    }
    
    void process_triangle_triple_point_indexed(YmmInt is_triangle, YmmInt triangle_index) {
        XmmFloatData const* triangles_ptr = (XmmFloatData const*)chunk.triangles;
        XmmFloat triangle_data[8];
        for (int i = 0; i < 8; ++i) {
            triangle_data[i] = XmmFloat(triangles_ptr[triangle_index.x[i]]);
        }
        YmmFloat ai_float, bi_float, ci_float, unused;
        XmmFloat::transpose8x4(triangle_data, ai_float, bi_float, ci_float, unused);
        YmmInt ai = YmmInt(ai_float);
        YmmInt bi = YmmInt(bi_float);
        YmmInt ci = YmmInt(ci_float);
        YmmInt aoffset = _mm256_slli_epi32(ai, 2);
        YmmInt boffset = _mm256_slli_epi32(bi, 2);
        YmmInt coffset = _mm256_slli_epi32(ci, 2);
        YmmVec3 a, b, c;
        a.x = _mm256_i32gather_ps((float const*)chunk.vertices,     aoffset, 4);
        a.y = _mm256_i32gather_ps((float const*)chunk.vertices + 1, aoffset, 4);
        a.z = _mm256_i32gather_ps((float const*)chunk.vertices + 2, aoffset, 4);
        b.x = _mm256_i32gather_ps((float const*)chunk.vertices,     boffset, 4);
        b.y = _mm256_i32gather_ps((float const*)chunk.vertices + 1, boffset, 4);
        b.z = _mm256_i32gather_ps((float const*)chunk.vertices + 2, boffset, 4);
        c.x = _mm256_i32gather_ps((float const*)chunk.vertices,     coffset, 4);
        c.y = _mm256_i32gather_ps((float const*)chunk.vertices + 1, coffset, 4);
        c.z = _mm256_i32gather_ps((float const*)chunk.vertices + 2, coffset, 4);
        process_triangle_triple_point_common(is_triangle, triangle_index, a, b, c);
    }
    
    void process_triangle_triple_point_immediate(YmmInt is_triangle, YmmInt triangle_index) {
        YmmFloatData const* triangles_ptr = (YmmFloatData const*)chunk.triangles;
        YmmFloat triangle_data_a[8];
        for (int i = 0; i < 8; ++i) {
            triangle_data_a[i] = YmmFloat(triangles_ptr[triangle_index.x[i] * 2]);
        }
        YmmVec3 a, b, c;
        YmmFloat::transpose8(triangle_data_a, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y);
        YmmInt czoffset = _mm256_slli_epi32(triangle_index, 4);
        c.z = _mm256_i32gather_ps((float const*)chunk.triangles + 8, czoffset, 4);
        process_triangle_triple_point_common(is_triangle, triangle_index, a, b, c);
    }
    
    void process_triangle_middle_coord_permuted(YmmBox box, YmmInt is_triangle, YmmInt triangle_index) {
        XmmFloatData const* triangles_ptr = (XmmFloatData const*)chunk.triangles;
        XmmFloat triangle_data[8];
        for (int i = 0; i < 8; ++i) {
            triangle_data[i] = XmmFloat(triangles_ptr[triangle_index.x[i]]);
        }
        YmmVec3 tmid;
        YmmFloat pmask_float;
        XmmFloat::transpose8x4(triangle_data, tmid.x, tmid.y, tmid.z, pmask_float);
        YmmInt pmask = YmmInt(pmask_float);
        YmmVec3 tmin, tmax;
        switch (pipeline.params.box_rep) {
        case RayPipelineParams::BoxRep::MinMax:
        {
            tmin = YmmVec3{box.v[0], box.v[1], box.v[2]};
            tmax = YmmVec3{box.v[3], box.v[4], box.v[5]};
            break;
        }
        case RayPipelineParams::BoxRep::CenterExtent:
        {
            YmmVec3 center = YmmVec3{box.v[0], box.v[1], box.v[2]};
            YmmVec3 extent = YmmVec3{box.v[3], box.v[4], box.v[5]};
            tmin = center - extent;
            tmax = center + extent;
            break;
        }
        }
        YmmVec3 a, b, c;
        a.x = YmmFloat::blend(YmmFloat::blend(tmin.x, tmid.x, YmmFloat(_mm256_slli_epi32(pmask, 31))), tmax.x, YmmFloat(_mm256_slli_epi32(pmask, 30)));
        a.y = YmmFloat::blend(YmmFloat::blend(tmin.y, tmid.y, YmmFloat(_mm256_slli_epi32(pmask, 29))), tmax.y, YmmFloat(_mm256_slli_epi32(pmask, 28)));
        a.z = YmmFloat::blend(YmmFloat::blend(tmin.z, tmid.z, YmmFloat(_mm256_slli_epi32(pmask, 27))), tmax.z, YmmFloat(_mm256_slli_epi32(pmask, 26)));
        b.x = YmmFloat::blend(YmmFloat::blend(tmin.x, tmid.x, YmmFloat(_mm256_slli_epi32(pmask, 25))), tmax.x, YmmFloat(_mm256_slli_epi32(pmask, 24)));
        b.y = YmmFloat::blend(YmmFloat::blend(tmin.y, tmid.y, YmmFloat(_mm256_slli_epi32(pmask, 23))), tmax.y, YmmFloat(_mm256_slli_epi32(pmask, 22)));
        b.z = YmmFloat::blend(YmmFloat::blend(tmin.z, tmid.z, YmmFloat(_mm256_slli_epi32(pmask, 21))), tmax.z, YmmFloat(_mm256_slli_epi32(pmask, 20)));
        c.x = YmmFloat::blend(YmmFloat::blend(tmin.x, tmid.x, YmmFloat(_mm256_slli_epi32(pmask, 19))), tmax.x, YmmFloat(_mm256_slli_epi32(pmask, 18)));
        c.y = YmmFloat::blend(YmmFloat::blend(tmin.y, tmid.y, YmmFloat(_mm256_slli_epi32(pmask, 17))), tmax.y, YmmFloat(_mm256_slli_epi32(pmask, 16)));
        c.z = YmmFloat::blend(YmmFloat::blend(tmin.z, tmid.z, YmmFloat(_mm256_slli_epi32(pmask, 15))), tmax.z, YmmFloat(_mm256_slli_epi32(pmask, 14)));
        process_triangle_triple_point_common(is_triangle, triangle_index, a, b, c);
    }

    void process_triangle_box_centered_common(YmmInt is_triangle, YmmInt triangle_index, YmmVec3 center, YmmVec3 du, YmmVec3 dv, YmmVec3 dn) {
        YmmFloat det = -dot(direction, dn);
        YmmFloat is_in_front = YmmFloat(is_triangle) & (det >= YmmFloat::set1(0));
        YmmVec3 origin_relative = origin - center;
        YmmFloat t = dot(origin_relative, dn);
        YmmFloat scaled_min_param = min_param * det;
        YmmFloat scaled_max_param = max_param * det;
        YmmFloat is_valid_param = is_in_front & (scaled_min_param <= t) & (t <= scaled_max_param);
        if (is_valid_param.any()) {
            YmmFloat param = t * det.recip();
            YmmVec3 rel_hit = origin_relative + param * direction;
            YmmFloat hit_u = dot(rel_hit, du);
            YmmFloat hit_v = dot(rel_hit, dv);
            YmmFloat is_interior = (hit_u >= YmmFloat::set1(-1)) & (hit_v >= YmmFloat::set1(-1)) & (hit_u + hit_v <= YmmFloat::set1(1));
            YmmFloat is_valid_hit = is_valid_param & is_interior;
            if (is_valid_hit.any()) {
                max_param = YmmFloat::blend(max_param, param, is_valid_hit);
                hit_triangle_index = YmmInt::blend(hit_triangle_index, triangle_index, YmmInt(is_valid_hit));
            }
        }
    }

    void process_triangle_box_centered_uv(YmmBox box, YmmInt is_triangle, YmmInt triangle_index) {
        YmmFloatData const* triangles_ptr = (YmmFloatData const*)chunk.triangles;
        YmmFloat triangle_data_a[8];
        for (int i = 0; i < 8; ++i) {
            triangle_data_a[i] = YmmFloat(triangles_ptr[triangle_index.x[i]]);
        }
        YmmVec3 du, dv;
        YmmFloat unused1, unused2;
        YmmFloat::transpose8(triangle_data_a, du.x, du.y, du.z, dv.x, dv.y, dv.z, unused1, unused2);
        YmmVec3 dn = cross(du, dv);
        YmmVec3 center;
        switch (pipeline.params.box_rep) {
        case RayPipelineParams::BoxRep::MinMax:
        {
            YmmVec3 tmin = YmmVec3{box.v[0], box.v[1], box.v[2]};
            YmmVec3 tmax = YmmVec3{box.v[3], box.v[4], box.v[5]};
            center = YmmFloat::set1(0.5f) * (tmin + tmax);
            break;
        }
        case RayPipelineParams::BoxRep::CenterExtent:
        {
            center = YmmVec3{box.v[0], box.v[1], box.v[2]};
            break;
        }
        }
        process_triangle_box_centered_common(is_triangle, triangle_index, center, du, dv, dn);
    }
    
    void process_triangle(YmmBox box, YmmInt is_triangle, YmmInt triangle_index) {
        switch (pipeline.params.triangle_rep) {
        case RayPipelineParams::TriangleRep::TriplePointIndexed:
            return process_triangle_triple_point_indexed(is_triangle, triangle_index);
        case RayPipelineParams::TriangleRep::TriplePointImmediate:
            return process_triangle_triple_point_immediate(is_triangle, triangle_index);
        case RayPipelineParams::TriangleRep::MiddleCoordPermuted:
            return process_triangle_middle_coord_permuted(box, is_triangle, triangle_index);
        case RayPipelineParams::TriangleRep::BoxCenteredUV:
            return process_triangle_box_centered_uv(box, is_triangle, triangle_index);
        case RayPipelineParams::TriangleRep::BoxCenteredUVExplicitN:
        {
            break;
        }
        }
    }

    void load_node_data(YmmInt node_index, YmmBox& box, YmmInt& subtree_size, YmmInt& triangle_index) {
        YmmIntData node_index_elems = node_index.data();
        YmmFloat node_data[8];
        for (int i = 0; i < 8; ++i) {
            node_data[i] = YmmFloat(((YmmFloatData const*)chunk.nodes)[node_index_elems.x[i]]);
        }
        YmmFloat subtree_size_float, triangle_index_float;
        YmmFloat::transpose8(
            node_data,
            box.v[0], box.v[1], box.v[2], box.v[3], box.v[4], box.v[5],
            subtree_size_float, triangle_index_float);
        subtree_size = YmmInt(subtree_size_float);
        triangle_index = YmmInt(triangle_index_float);
    }

    void process() {
        ray_space_shear = YmmSpaceShear::make_for(direction);
        YmmInt sentinel_node_index = YmmInt::set1(chunk.sentinel_node_index);
        YmmInt node_index = YmmInt::set1(0);
        while (true) {
            YmmInt is_valid_node = node_index < sentinel_node_index;
            if (is_valid_node.none_bit()) {
                return;
            }
            YmmBox box;
            YmmInt subtree_size, triangle_index;
            load_node_data(node_index, box, subtree_size, triangle_index);
            YmmFloat minp = min_param;
            YmmFloat maxp = max_param;
            clip_param_interval(box, minp, maxp);
            YmmFloat bbox_overlaps = minp <= maxp;
            if (triangle_index.any_bit()) {
                YmmInt is_triangle = triangle_index > YmmInt::set1(0);
                process_triangle(box, is_triangle, triangle_index);
            }
            YmmInt node_index_delta = YmmInt::blend(subtree_size, YmmInt::set1(1), YmmInt(bbox_overlaps));
            node_index = node_index + node_index_delta;
        }
    }
};
/*
struct RayPipeline::Internal::ChunkIntersectorBulk {
    RayPipeline& pipeline;
    Chunk& chunk;

    void process_buffer(std::unique_ptr<RayBuffer2> buffer) {
        //for (std::unique_ptr<Ray>& ray_ptr : *buffer) {
        //    Ray& ray = *ray_ptr;
        //    Internal::ChunkIntersectorImmediate chunk_intersector{
        //        pipeline.params,
        //        chunk,
        //        ray._origin,
        //        ray._direction,
        //        ray._dir_inverse,
        //        ray._space_shear,
        //        ray._min_param,
        //        ray._max_param,
        //    };
        //    chunk_intersector.intersect_all();
        //    if (chunk_intersector.hit_triangle_index > 0) {
        //        ray._min_param = chunk_intersector.ray_min_param;
        //        ray._max_param = chunk_intersector.ray_max_param;
        //        ray._hit_triangle_index = chunk_intersector.chunk.triangle_index_offset + (size_t)chunk_intersector.hit_triangle_index;
        //        ray._hit_triangle_pos = chunk_intersector.hit_triangle_pos;
        //    }
        //    RayScheduler{pipeline}.dispatch_ray(std::move(ray_ptr));
        //}
        size_t ray_count = buffer->size();
        size_t batch_count = ray_count / 8;
        for (size_t batch_i = 0; batch_i < batch_count; ++batch_i) {
            RayPipeline::Internal::ChunkIntersectorSimd intersector_simd{pipeline, chunk};
            for (size_t j = 0; j < 8; ++j) {
                Ray& ray = *(*buffer)[batch_i * 8 + j];
                intersector_simd.origin.x.x[j] = ray._origin.x;
                intersector_simd.origin.y.x[j] = ray._origin.y;
                intersector_simd.origin.z.x[j] = ray._origin.z;
                intersector_simd.direction.x.x[j] = ray._direction.x;
                intersector_simd.direction.y.x[j] = ray._direction.y;
                intersector_simd.direction.z.x[j] = ray._direction.z;
                intersector_simd.dir_inverse.x.x[j] = ray._dir_inverse.x;
                intersector_simd.dir_inverse.y.x[j] = ray._dir_inverse.y;
                intersector_simd.dir_inverse.z.x[j] = ray._dir_inverse.z;
                intersector_simd.min_param.x[j] = ray._min_param;
                intersector_simd.max_param.x[j] = ray._max_param;
                intersector_simd.hit_triangle_index.x[j] = 0;
            }
            intersector_simd.process();
            for (size_t j = 0; j < 8; ++j) {
                Ray& ray = *(*buffer)[batch_i * 8 + j];
                if (intersector_simd.hit_triangle_index.x[j] > 0) {
                    ray._max_param = intersector_simd.max_param.x[j];
                    ray._hit_triangle_index = chunk.triangle_index_offset + (size_t)intersector_simd.hit_triangle_index.x[j];
                }
                RayScheduler{pipeline}.dispatch_ray(std::move((*buffer)[batch_i * 8 + j]));
            }
        }
        size_t leftover_offset = batch_count * 8;
        if (leftover_offset < ray_count) {
            size_t leftover_count = ray_count - leftover_offset;
            RayPipeline::Internal::ChunkIntersectorSimd intersector_simd{pipeline, chunk};
            for (size_t j = 0; j < leftover_count; ++j) {
                Ray& ray = *(*buffer)[leftover_offset + j];
                intersector_simd.origin.x.x[j] = ray._origin.x;
                intersector_simd.origin.y.x[j] = ray._origin.y;
                intersector_simd.origin.z.x[j] = ray._origin.z;
                intersector_simd.direction.x.x[j] = ray._direction.x;
                intersector_simd.direction.y.x[j] = ray._direction.y;
                intersector_simd.direction.z.x[j] = ray._direction.z;
                intersector_simd.dir_inverse.x.x[j] = ray._dir_inverse.x;
                intersector_simd.dir_inverse.y.x[j] = ray._dir_inverse.y;
                intersector_simd.dir_inverse.z.x[j] = ray._dir_inverse.z;
                intersector_simd.min_param.x[j] = ray._min_param;
                intersector_simd.max_param.x[j] = ray._max_param;
                intersector_simd.hit_triangle_index.x[j] = 0;
            }
            for (size_t j = leftover_count; j < 8; ++j) {
                intersector_simd.origin.x.x[j] = NAN;
                intersector_simd.origin.y.x[j] = NAN;
                intersector_simd.origin.z.x[j] = NAN;
                intersector_simd.direction.x.x[j] = NAN;
                intersector_simd.direction.y.x[j] = NAN;
                intersector_simd.direction.z.x[j] = NAN;
                intersector_simd.dir_inverse.x.x[j] = NAN;
                intersector_simd.dir_inverse.y.x[j] = NAN;
                intersector_simd.dir_inverse.z.x[j] = NAN;
                intersector_simd.min_param.x[j] = HUGE_VALF;
                intersector_simd.max_param.x[j] = -HUGE_VALF;
                intersector_simd.hit_triangle_index.x[j] = 0;
            }
            intersector_simd.process();
            for (size_t j = 0; j < leftover_count; ++j) {
                Ray& ray = *(*buffer)[leftover_offset + j];
                if (intersector_simd.hit_triangle_index.x[j] > 0) {
                    ray._max_param = intersector_simd.max_param.x[j];
                    ray._hit_triangle_index = chunk.triangle_index_offset + (size_t)intersector_simd.hit_triangle_index.x[j];
                }
                RayScheduler{pipeline}.dispatch_ray(std::move((*buffer)[leftover_offset + j]));
            }
        }
    }
};
*/
/*
void RayPipeline::Internal::RayScheduler::dispatch_ray(std::unique_ptr<Ray> ray_ptr) {
    Tree& ptree = *pipeline.zone_trees[ray_ptr->_zone_index];
    size_t total_chunk_count = ptree.chunks.size();
    while (!ray_ptr->_reservation_min_heap.empty()) {
        Ray::NodeReservation min_res = ray_ptr->pop_min_reservation();
        if (min_res.min_hit_param > ray_ptr->_max_param) {
            ray_ptr->_reservation_min_heap.clear();
            break;
        }

        size_t node_index = min_res.node_index;

        if (node_index < total_chunk_count) {
            Chunk& chunk = *ptree.chunks[node_index];
            if (!chunk.ray_buffer_current) {
                chunk.ray_buffer_current = pipeline.create_ray_buffer_2();
            }
            chunk.ray_buffer_current->push_back(std::move(ray_ptr));
            if (chunk.ray_buffer_current->size() >= pipeline.params.ray_buffer_size) {
                ChunkIntersectorBulk{pipeline, chunk}.process_buffer(std::move(chunk.ray_buffer_current));
            }
            return;
        } else {
            size_t children_begin = ptree.node_child_indexes[node_index - total_chunk_count];
            size_t children_end = ptree.node_child_indexes[node_index - total_chunk_count + 1];
            for (size_t i = children_begin; i < children_end; ++i) {
                float minp = ray_ptr->_min_param, maxp = ray_ptr->_max_param;
                Utils::clip_param_interval(pipeline.params, ray_ptr->_origin, ray_ptr->_dir_inverse, ptree.node_boxes[i], minp, maxp);
                if (minp <= maxp) {
                    ray_ptr->insert_reservation(i, minp);
                }
            }
        }
    }
    ray_ptr->on_completed(ray_ptr, pipeline);
}
*/
/*
void RayPipeline::schedule(std::unique_ptr<Ray> ray_ptr) {
    if (ray_ptr->_zone_index >= zone_trees.size()) {
        return ray_ptr->on_completed(ray_ptr, *this);
    }
    ray_ptr->_reservation_min_heap.clear();

    Tree const& ptree = *zone_trees[ray_ptr->_zone_index];
    size_t root_index = ptree.node_boxes.size() - 1;

    float minp = ray_ptr->_min_param, maxp = ray_ptr->_max_param;
    Internal::Utils::clip_param_interval(params, ray_ptr->_origin, ray_ptr->_dir_inverse, ptree.node_boxes[root_index], minp, maxp);
    if (minp <= maxp) {
        ray_ptr->insert_reservation(root_index, minp);
    }

    Internal::RayScheduler{*this}.dispatch_ray(std::move(ray_ptr));
}
*/
/*
void RayPipeline::flush() {
    bool done = false;
    while (!done) {
        done = true;
        for (auto& ptree_ptr : zone_trees) {
            for (auto& chunk_ptr : ptree_ptr->chunks) {
                if (chunk_ptr->ray_buffer_current) {
                    done = false;
                    Internal::ChunkIntersectorBulk{*this, *chunk_ptr}.process_buffer(std::move(chunk_ptr->ray_buffer_current));
                }
            }
        }
    }
}
*/

/*
void RayPipeline::schedule(Tree& ptree, RayData ray_data) {
    if (!ray_data.reservation_heap.empty()) {
        ptree.pending_funnel.insert(*this, std::move(ray_data));
    } else {
        ptree.completed_funnel.insert(*this, std::move(ray_data));
    }
}
*/

void RayPipeline::Internal::append_ray(RayPipeline& pipeline, size_t zone_id, std::unique_ptr<RayPacket>& packet_ptr, RayPacketFunnel& target_funnel, RayData ray_data) {
    if (!packet_ptr) {
        packet_ptr = pipeline.create_ray_packet(zone_id);
    }
    packet_ptr->set_ray_data(packet_ptr->ray_count, std::move(ray_data));
    packet_ptr->ray_count += 1;
    if (packet_ptr->ray_count == packet_ptr->capacity) {
        target_funnel.insert(std::move(packet_ptr));
    }
}

RayPipeline::Inserter RayPipeline::inserter(size_t zone_id) {
    if (zone_id < zone_trees.size()) {
        Tree& ptree = *zone_trees[zone_id];
        return Inserter(*this, ptree);
    } else {
        throw std::runtime_error("invalid zone id");
    }
}

RayPipeline::Inserter::Inserter(RayPipeline& pipeline, Tree& ptree) {
    pipeline_ptr = &pipeline;
    ptree_ptr = &ptree;
}

RayPipeline::Inserter::~Inserter() {
    if (packet_ptr) {
        if (packet_ptr->ray_count > 0) {
            ptree_ptr->pending_funnel.insert(std::move(packet_ptr));
        }
    }
}

void RayPipeline::Inserter::schedule(Vec3 origin, Vec3 direction, uint64_t extra_data, float min_param, float max_param) {
    RayData ray_data;
    ray_data.origin = origin;
    ray_data.direction = direction;
    ray_data.min_param = min_param;
    ray_data.max_param = max_param;
    ray_data.extra_data = extra_data;
    ray_data.hit_world_triangle_index = World::invalid_index;
    size_t root_index = ptree_ptr->node_boxes.size() - 1;
    float minp = min_param, maxp = max_param;
    Internal::Utils::clip_param_interval(pipeline_ptr->params, origin, elementwise_inverse(direction), ptree_ptr->node_boxes[root_index], minp, maxp);
    if (minp <= maxp) {
        ray_data.reservation_heap.insert_reservation(root_index, minp);
    }
    Internal::append_ray(*pipeline_ptr, zone_id, packet_ptr, ptree_ptr->pending_funnel, std::move(ray_data));
}

void RayPipeline::Inserter::schedule(Vec3 origin, Vec3 direction, uint64_t extra_data) {
    return schedule(origin, direction, extra_data, 0.0f, HUGE_VALF);
}

bool RayPipeline::completed_packets_empty() {
    std::lock_guard<std::mutex> guard(ray_packets_completed_mutex);
    return ray_packets_completed.empty();
}

std::unique_ptr<RayPipeline::RayPacket> RayPipeline::pop_completed_packet() {
    std::lock_guard<std::mutex> guard(ray_packets_completed_mutex);
    if (ray_packets_completed.empty()) {
        return nullptr;
    } else {
        std::unique_ptr<RayPacket> result = std::move(ray_packets_completed.back());
        ray_packets_completed.pop_back();
        return result;
    }
}

size_t RayPipeline::get_total_packet_count() {
    return total_packet_count.load(std::memory_order_relaxed);
}

struct RayPipeline::Internal::ChunkProcessor: Processor {
    RayPipeline& pipeline;
    RayPipeline::Tree& ptree;
    RayPipeline::Chunk& chunk;

    ChunkProcessor(RayPipeline& pipeline, RayPipeline::Tree& ptree, RayPipeline::Chunk& chunk);
    virtual bool has_pending_work_impl() override;
    virtual bool work_impl() override;
};

RayPipeline::Internal::ChunkProcessor::ChunkProcessor(RayPipeline& pipeline, RayPipeline::Tree& ptree, RayPipeline::Chunk& chunk)
    : Processor(pipeline.processor_control), pipeline(pipeline), ptree(ptree), chunk(chunk)
{
}

bool RayPipeline::Internal::ChunkProcessor::has_pending_work_impl() {
    return !chunk.pending_funnel.full_packets_empty();
}

bool RayPipeline::Internal::ChunkProcessor::work_impl() {
    bool work_performed = false;
    while (true) {
        std::unique_ptr<RayPacket> packet_ptr = chunk.pending_funnel.pop_full_packet();
        if (!packet_ptr) {
            break;
        }
        work_performed = true;
        size_t ray_count = packet_ptr->ray_count;
        size_t batch_count = (ray_count + 7) / 8;
        for (size_t i = ray_count; i < batch_count * 8; ++i) {
            packet_ptr->origin_x_array[i] = NAN;
            packet_ptr->origin_y_array[i] = NAN;
            packet_ptr->origin_z_array[i] = NAN;
            packet_ptr->direction_x_array[i] = NAN;
            packet_ptr->direction_y_array[i] = NAN;
            packet_ptr->direction_z_array[i] = NAN;
            packet_ptr->min_param_array[i] = HUGE_VALF;
            packet_ptr->max_param_array[i] = -HUGE_VALF;
        }
        for (size_t batch_i = 0; batch_i < batch_count; ++batch_i) {
            RayPipeline::Internal::ChunkIntersectorSimd intersector_simd{pipeline, chunk};
            intersector_simd.origin.x = YmmFloat(((YmmFloatData*)packet_ptr->origin_x_array)[batch_i]);
            intersector_simd.origin.y = YmmFloat(((YmmFloatData*)packet_ptr->origin_y_array)[batch_i]);
            intersector_simd.origin.z = YmmFloat(((YmmFloatData*)packet_ptr->origin_z_array)[batch_i]);
            intersector_simd.direction.x = YmmFloat(((YmmFloatData*)packet_ptr->direction_x_array)[batch_i]);
            intersector_simd.direction.y = YmmFloat(((YmmFloatData*)packet_ptr->direction_y_array)[batch_i]);
            intersector_simd.direction.z = YmmFloat(((YmmFloatData*)packet_ptr->direction_z_array)[batch_i]);
            intersector_simd.dir_inverse.x = intersector_simd.direction.x.recip();
            intersector_simd.dir_inverse.y = intersector_simd.direction.y.recip();
            intersector_simd.dir_inverse.z = intersector_simd.direction.z.recip();
            intersector_simd.min_param = YmmFloat(((YmmFloatData*)packet_ptr->min_param_array)[batch_i]);
            intersector_simd.max_param = YmmFloat(((YmmFloatData*)packet_ptr->max_param_array)[batch_i]);
            intersector_simd.hit_triangle_index = YmmInt::set1(0);
            intersector_simd.process();
            ((YmmFloatData*)packet_ptr->max_param_array)[batch_i] = intersector_simd.max_param.data();
            YmmIntData hit_chunk_triangle_index = intersector_simd.hit_triangle_index.data();
            for (size_t i = 0; i < 8; ++i) {
                if (hit_chunk_triangle_index.x[i] != 0) {
                    packet_ptr->hit_world_triangle_index_array[batch_i * 8 + i] = hit_chunk_triangle_index.x[i] + chunk.triangle_index_offset;
                }
            }
        }
        ptree.pending_funnel.insert(std::move(packet_ptr));
    }
    return work_performed;
}

struct RayPipeline::Internal::TreeProcessor: Processor {
    RayPipeline& pipeline;
    RayPipeline::Tree& ptree;
    size_t zone_id;

    TreeProcessor(RayPipeline& pipeline, size_t zone_id);
    virtual bool has_pending_work_impl() override;
    virtual bool work_impl() override;
};

RayPipeline::Internal::TreeProcessor::TreeProcessor(RayPipeline& pipeline, size_t zone_id)
    : Processor(pipeline.processor_control), pipeline(pipeline), ptree(*pipeline.zone_trees[zone_id]), zone_id(zone_id) {
}

bool RayPipeline::Internal::TreeProcessor::has_pending_work_impl() {
    return !ptree.pending_funnel.full_packets_empty() || !ptree.completed_funnel.full_packets_empty();
}

bool RayPipeline::Internal::TreeProcessor::work_impl() {
    bool work_performed = false;
    {
        size_t total_chunk_count = ptree.chunks.size();
        std::vector<std::unique_ptr<RayPacket>> per_chunk_buffers(total_chunk_count);
        std::unique_ptr<RayPacket> completed_buffer;
        while (true) {
            std::unique_ptr<RayPacket> packet_ptr = ptree.pending_funnel.pop_full_packet();
            if (!packet_ptr) {
                break;
            }
            work_performed = true;
            for (size_t i = 0; i < packet_ptr->ray_count; ++i) {
                RayData ray_data = packet_ptr->extract_ray_data(i);
                Vec3 dir_inverse = elementwise_inverse(ray_data.direction);
                size_t chunk_index_to_insert = World::invalid_index;
                while (!ray_data.reservation_heap.empty()) {
                    ReservationHeap::NodeReservation min_res = ray_data.reservation_heap.pop_min_reservation();
                    if (min_res.min_hit_param > ray_data.max_param) {
                        ray_data.reservation_heap.clear();
                        break;
                    }
                    size_t node_index = min_res.node_index;
                    if (node_index < total_chunk_count) {
                        chunk_index_to_insert = node_index;
                        break;
                    } else {
                        size_t children_begin = ptree.node_child_indexes[node_index - total_chunk_count];
                        size_t children_end = ptree.node_child_indexes[node_index - total_chunk_count + 1];
                        for (size_t i = children_begin; i < children_end; ++i) {
                            float minp = ray_data.min_param, maxp = ray_data.max_param;
                            Utils::clip_param_interval(pipeline.params, ray_data.origin, dir_inverse, ptree.node_boxes[i], minp, maxp);
                            if (minp <= maxp) {
                                ray_data.reservation_heap.insert_reservation(i, minp);
                            }
                        }
                    }
                }
                if (chunk_index_to_insert != World::invalid_index) {
                    Internal::append_ray(pipeline, zone_id, per_chunk_buffers[chunk_index_to_insert], ptree.chunks[chunk_index_to_insert]->pending_funnel, std::move(ray_data));
                } else {
                    Internal::append_ray(pipeline, zone_id, completed_buffer, ptree.completed_funnel, std::move(ray_data));
                }
            }
            pipeline.collect_ray_packet_spare(std::move(packet_ptr));
        }
        for (size_t chunk_index = 0; chunk_index < total_chunk_count; ++chunk_index) {
            if (per_chunk_buffers[chunk_index] && per_chunk_buffers[chunk_index]->ray_count > 0) {
                ptree.chunks[chunk_index]->pending_funnel.insert(std::move(per_chunk_buffers[chunk_index]));
            }
        }
        if (completed_buffer && completed_buffer->ray_count > 0) {
            ptree.completed_funnel.insert(std::move(completed_buffer));
        }
    }
    while (std::unique_ptr<RayPacket> completed_packet_ptr = ptree.completed_funnel.pop_full_packet()) {
        work_performed = true;
        std::lock_guard<std::mutex> guard(pipeline.ray_packets_completed_mutex);
        pipeline.ray_packets_completed.push_back(std::move(completed_packet_ptr));
    }
    return work_performed;
}

void RayPipeline::Internal::Loader::initialize_processor_objects() {
    for (size_t zone_id = 0; zone_id < pipeline.zone_trees.size(); ++zone_id) {
        std::unique_ptr<Tree>& ptree_ptr = pipeline.zone_trees[zone_id];
        Scheduler::register_processor(std::make_shared<TreeProcessor>(pipeline, zone_id));
        for (std::unique_ptr<Chunk>& chunk_ptr : ptree_ptr->chunks) {
            Scheduler::register_processor(std::make_shared<ChunkProcessor>(pipeline, *ptree_ptr, *chunk_ptr));
        }
    }
}
