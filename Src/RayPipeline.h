#pragma once

#include "AlignedBuffer.h"
#include "Threading.h"
#include "Vec3.h"
#include "World.h"
#include "Ymm.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include <immintrin.h>

struct RayPipelineParams {
    enum class BoxRep {
        MinMax,
        CenterExtent,
    };

    enum class TriangleRep {
        TriplePointIndexed,
        TriplePointImmediate,
        BoxCenteredUV,
        BoxCenteredUVExplicitN,
        MiddleCoordPermuted,
    };

    int max_chunk_height = 8;
    int ray_buffer_size = 0x100;
    BoxRep box_rep = BoxRep::MinMax;
    TriangleRep triangle_rep = TriangleRep::TriplePointIndexed;

    size_t sizeof_packed_triangle() const;
    size_t alignof_packed_triangle() const;
};

class Ray {
private:
    struct NodeReservation {
        size_t node_index;
        float min_hit_param;

        static bool compare_node_reservation_greater(NodeReservation const& a, NodeReservation const& b);
    };

    struct SpaceShear {
        int kx, ky, kz;
        float sx, sy, sz;
        //int rots;
        //bool refl;

        static SpaceShear make_for(Vec3 dir);
    };

    size_t _zone_index;
    Vec3 _origin, _direction, _dir_inverse;
    SpaceShear _space_shear;
    float _min_param, _max_param;
    size_t _hit_triangle_index;
    Vec3 _hit_triangle_pos;
    std::vector<NodeReservation> _reservation_min_heap;

    NodeReservation pop_min_reservation();
    void insert_reservation(size_t node_index, float min_hit_param);

public:
    friend struct RayPipeline;

    Ray(size_t zone_index, Vec3 origin, Vec3 direction, float min_param = 0.0f, float max_param = HUGE_VALF);
    Ray(Ray const&) = default;
    Ray(Ray&&) = default;
    virtual ~Ray();
    Ray& operator=(Ray const&) = default;
    Ray& operator=(Ray&&) = default;

    void reset(size_t zone_index, float min_param = 0.0f, float max_param = HUGE_VALF);
    void reset(Vec3 direction, float min_param = 0.0f, float max_param = HUGE_VALF);
    void reset(size_t zone_index, Vec3 direction, float min_param = 0.0f, float max_param = HUGE_VALF);
    void reset(Vec3 origin, Vec3 direction, float min_param = 0.0f, float max_param = HUGE_VALF);
    void reset(size_t zone_index, Vec3 origin, Vec3 direction, float min_param = 0.0f, float max_param = HUGE_VALF);

    size_t zone_index() const { return _zone_index; }
    Vec3 origin() const { return _origin; }
    Vec3 direction() const { return _direction; }
    float min_param() const { return _min_param; }
    float max_param() const { return _max_param; }
    size_t hit_triangle_index() const { return _hit_triangle_index; }
    Vec3 hit_triangle_pos() const { return _hit_triangle_pos; }
    Vec3 hit_point() const { return _origin + _max_param * _direction; }

    virtual void on_completed(std::unique_ptr<Ray>& self_ptr, RayPipeline& pipeline);
};

struct RayPipeline {
private:
    struct Internal;

    struct alignas(16) AlignedVec3 {
        float x, y, z;

        operator Vec3() const {
            return Vec3{x, y, z};
        }
    };

    struct alignas(16) PackedTriangleTriplePointIndexed {
        int vertex_indexes[3];
    };

    struct alignas(32) PackedTriangleTriplePointImmediate {
        Vec3 vertices[3];
    };

    struct alignas(32) PackedTriangleBoxCenteredUV {
        Vec3 du, dv;
    };

    struct alignas(32) PackedTriangleBoxCenteredUVExplicitN {
        Vec3 du, dv, dn;
    };

    struct alignas(16) PackedTriangleMiddleCoordPermuted {
        Vec3 mid;
        uint32_t permutation_mask;
    };

    struct MinMaxBox {
        Vec3 min;
        Vec3 max;
    };

    struct CenterExtentBox {
        Vec3 center;
        Vec3 extent;
    };

    union Box {
        MinMaxBox as_minmax;
        CenterExtentBox as_center_extent;
    };

    struct alignas(32) PackedNode {
        Box box;
        int subtree_size;
        int triangle_index;
    };

    struct ReservationHeap {
        struct NodeReservation {
            size_t node_index;
            float min_hit_param;

            static bool compare_greater(NodeReservation const& a, NodeReservation const& b);
        };

        std::vector<NodeReservation> heap_data;

        ReservationHeap() = default;
        ReservationHeap(ReservationHeap&&) = default;
        ~ReservationHeap() = default;
        ReservationHeap& operator=(ReservationHeap&&) = default;

        NodeReservation pop_min_reservation();
        void insert_reservation(size_t node_index, float min_hit_param);
        bool empty();
        void clear();
    };

    //using RayBuffer2 = std::vector<std::unique_ptr<Ray>>;

public:
    struct RayData {
        Vec3 origin, direction;
        float min_param, max_param;
        ReservationHeap reservation_heap;
        size_t hit_world_triangle_index;
        uint64_t extra_data;
    };

    struct RayPacket {
        AlignedBuffer buffer;
        int capacity, ray_count;
        float* origin_x_array;
        float* origin_y_array;
        float* origin_z_array;
        float* direction_x_array;
        float* direction_y_array;
        float* direction_z_array;
        float* min_param_array;
        float* max_param_array;
        ReservationHeap* reservation_heap_array;
        size_t* hit_world_triangle_index_array;
        uint64_t* extra_data_array;
        uint64_t id;
        size_t zone_id;

        void set_ray_data(size_t index, RayData ray_data);
        RayData extract_ray_data(size_t index);
    };

private:
    struct RayPacketFunnel {
        std::mutex full_packets_mutex;
        std::vector<std::unique_ptr<RayPacket>> full_packets;

        void insert(std::unique_ptr<RayPacket> ray_packet);
        bool full_packets_empty();
        std::unique_ptr<RayPacket> pop_full_packet();
    };

    struct Chunk {
        AlignedBuffer buffer;
        int vertex_count, triangle_count, sentinel_node_index;
        ptrdiff_t triangle_index_offset;
        AlignedVec3* vertices;
        void* triangles;
        PackedNode* nodes;
        //std::vector<std::unique_ptr<RayBuffer2>> ray_buffers_pending;
        //std::unique_ptr<RayBuffer2> ray_buffer_current;
        RayPacketFunnel pending_funnel;
    };

    struct Tree {
        size_t index;
        std::vector<std::unique_ptr<Chunk>> chunks;
        std::vector<Box> node_boxes;
        std::vector<size_t> node_child_indexes;
        RayPacketFunnel pending_funnel;
        RayPacketFunnel completed_funnel;
    };

public:
    struct Inserter {
        RayPipeline* pipeline_ptr;
        Tree* ptree_ptr;
        size_t zone_id;
        std::unique_ptr<RayPacket> packet_ptr;

        Inserter(RayPipeline& pipeline, Tree& ptree);
        Inserter(Inserter&&) = default;
        Inserter& operator=(Inserter&&) = default;
        ~Inserter();
        void schedule(Vec3 origin, Vec3 direction, uint64_t extra_data, float min_param, float max_param);
        void schedule(Vec3 origin, Vec3 direction, uint64_t extra_data);
    };

private:
    RayPipelineParams params;
    std::vector<std::unique_ptr<Tree>> zone_trees;
    std::shared_ptr<ProcessorControl> processor_control;
    std::mutex ray_packets_spare_mutex;
    std::vector<std::unique_ptr<RayPacket>> ray_packets_spare;
    std::mutex ray_packets_completed_mutex;
    std::vector<std::unique_ptr<RayPacket>> ray_packets_completed;
    std::atomic<size_t> total_packet_count;
    std::atomic<uint64_t> next_packet_id;

public:
    friend struct RayPipelineParams;

    RayPipeline(World const& world, RayPipelineParams const& params);
    ~RayPipeline();

    RayPipelineParams const& get_params();

    std::unique_ptr<RayPacket> create_ray_packet(size_t zone_id);
    void collect_ray_packet_spare(std::unique_ptr<RayPacket> packet_ptr);
    //std::unique_ptr<RayBuffer2> create_ray_buffer_2();

    void trace_immediately(Ray& ray);
    //void schedule(std::unique_ptr<Ray> ray_ptr);
    //void flush();

    //void schedule(Tree& ptree, RayData ray_data);
    //void schedule(size_t zone_id, Vec3 origin, Vec3 direction, void* extra_data, float min_param, float max_param);
    //void schedule(size_t zone_id, Vec3 origin, Vec3 direction, void* extra_data);

    Inserter inserter(size_t zone_id);

    bool completed_packets_empty();
    std::unique_ptr<RayPacket> pop_completed_packet();

    size_t get_total_packet_count();
};
