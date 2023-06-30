#include "RendererThread.h"
#include "ParameterSampler.h"
#include "ForwardSampler.h"
#include "BidirSampler.h"
#include "World.h"
#include "RayPipeline.h"
#include "Threading.h"
//#include <png.h>

constexpr int width = 1600;
constexpr int height = 900;
//constexpr float exposure = 0.5f;
constexpr float exposure = 70.0f;
constexpr float accum_num_scale = 10000000.0f;
constexpr float accum_den_scale = 10.0f * width * height;

constexpr float pi = 3.141593f;

static size_t ceil_log2(size_t x) {
    if (x > ((size_t)(-1) >> 1)) {
        throw std::runtime_error("rounding up to a power of 2 would result in overflow");
    }
    size_t r = 1;
    while (r < x) {
        r <<= 1;
    }
    return r;
}

static uint8_t tonemap_curve(float x) {
    float t;
    float v = x * exposure;
    //t = 1.0f - expf(-v);
    t = v / (v + 1.0f);
    t = fminf(fmaxf(t, 0), 1);
    t = sqrtf(t);
    int b = (int)(t * 255.0f);
    return b;
}

static void tonemap(float va, float vb, float vc, uint8_t* pixel)
{
    pixel[0] = tonemap_curve(vc);
    pixel[1] = tonemap_curve(vb);
    pixel[2] = tonemap_curve(va);
    pixel[3] = 255;
}

struct Random {
    uint64_t state = 0x4d595df4d0f33173u;
    static constexpr uint64_t multiplier = 6364136223846793005u;
    static constexpr uint64_t increment = 1442695040888963407u;

    static uint32_t rotr32(uint32_t x, unsigned r) {
        return x >> r | x << ((32 - r) & 31);
    }

    void uniform(uint32_t& r) {
        uint64_t x = state;
        unsigned count = (unsigned)(x >> 59);
        state = x * multiplier + increment;
        x ^= x >> 18;
        r = rotr32((uint32_t)(x >> 27), count);
    }

    void uniform(float& q) {
        uint32_t u;
        uniform(u);
        q = 0x1p-32f * (float)u;
    }

    void triangle(float& u) {
        float q1, q2;
        uniform(q1);
        uniform(q2);
        u = q1 - q2;
    }

    /* normal distribution with stddev 1 */
    void gauss(float& u) {
        float q;
        u = 0;
        for (int i = 0; i < 6*9; ++i) {
            triangle(q);
            u += q;
        }
        u *= (1.0f / 3.0f);
    }

    void circle(float& u, float& v) {
        while (true) {
            uniform(u);
            u = u * 2.0f - 1.0f;
            uniform(v);
            v = v * 2.0f - 1.0f;
            float uvsqr = u * u + v * v;
            if (uvsqr > 1.0f) {
                continue;
            }
            float inv = 1.0f / sqrtf(uvsqr);
            u *= inv;
            v *= inv;
            return;
        }
    }

    void lambert(Vec3 const& n, Vec3& d) {
        Vec3 a{1, 0, 0};
        if (n.x > 0.8f || n.x < -0.8f) {
            a = Vec3{0, 1, 0};
        }
        Vec3 b = norm(cross(a, n));
        Vec3 t = norm(cross(b, n));
        float q;
        uniform(q);
        float z = sqrtf(q);
        float r = sqrtf(1.0f - q);
        float u;
        float v;
        circle(u, v);
        u *= r;
        v *= r;
        d = z * n + u * b + v * t;
    }

    void p(float beta, bool& r) {
        float q;
        uniform(q);
        r = q < beta;
    }
};

struct DiscreteDistribution {
    struct Elem {
        size_t alt;
        uint32_t threshold;
    };

    std::vector<Elem> elems;

    static DiscreteDistribution make(std::vector<float> const& input_weights) {
        constexpr uint64_t cp_unit = (uint64_t)1 << 32;
        size_t size = ceil_log2(input_weights.size());
        float total_weight = 0.0f;
        for (size_t i = 0; i < input_weights.size(); ++i) {
            total_weight += input_weights[i];
        }
        std::vector<uint64_t> scaled_commit_probabilities(size);
        std::vector<size_t> overfull_indices;
        std::vector<size_t> underfull_indices;
        float probability_scale = (float)cp_unit * (float)size / total_weight;
        for (size_t i = 0; i < input_weights.size(); ++i) {
            scaled_commit_probabilities[i] = probability_scale * input_weights[i];
            if (scaled_commit_probabilities[i] < cp_unit) {
                underfull_indices.push_back(i);
            } else {
                if (scaled_commit_probabilities[i] > cp_unit) {
                    overfull_indices.push_back(i);
                }
            }
        }
        for (size_t i = input_weights.size(); i < size; ++i) {
            scaled_commit_probabilities[i] = 0;
            underfull_indices.push_back(i);
        }
        std::vector<Elem> elems(size);
        for (size_t i = 0; i < size; ++i) {
            elems[i].alt = i;
            elems[i].threshold = 0xffffffff;
        }
        while (!overfull_indices.empty() && !underfull_indices.empty()) {
            size_t next_overfull = overfull_indices.back();
            size_t next_underfull = underfull_indices.back();
            underfull_indices.pop_back();
            uint64_t delta = cp_unit - scaled_commit_probabilities[next_underfull];
            scaled_commit_probabilities[next_overfull] -= delta;
            elems[next_underfull].alt = next_overfull;
            elems[next_underfull].threshold = cp_unit - delta;
            if (scaled_commit_probabilities[next_overfull] <= cp_unit) {
                overfull_indices.pop_back();
                if (scaled_commit_probabilities[next_overfull] < cp_unit) {
                    underfull_indices.push_back(next_overfull);
                }
            }
        }
        return DiscreteDistribution{std::move(elems)};
    }

    void sample(Random& random, size_t& result) {
        uint32_t qi, qa;
        random.uniform(qi);
        result = qi & (elems.size() - 1);
        random.uniform(qa);
        if (qa >= elems[result].threshold) {
            result = elems[result].alt;
        }
    }
};

struct Material {
    Vec3 albedo = Vec3{0, 0, 0};
    Vec3 emission = Vec3{0, 0, 0};

    static Material emissive(float ex, float ey, float ez) {
        Material m;
        m.emission = Vec3{ex, ey, ez};
        return m;
    }

    static Material diffuse(float ax, float ay, float az) {
        Material m;
        m.albedo = Vec3{ax, ay, az};
        return m;
    }

    Vec3 emission_flux_density(Vec3 normal, Vec3 lens_dir) {
        return emission;
    }

    float emission_probability_density(Vec3 normal, Vec3 lens_dir) {
        return 1.0f / pi;
    }

    void sample_emission(Random& random, Vec3 normal, Vec3& light_dir, Vec3& result_sample) {
        random.lambert(normal, light_dir);
        result_sample = pi * emission;
    }

    float scatter_event_probability_from_lens(Vec3 normal, Vec3 lens_dir) {
        return fmaxf(fmaxf(albedo.x, albedo.y), albedo.z);
    }

    float scatter_event_probability_from_light(Vec3 normal, Vec3 lens_dir) {
        return scatter_event_probability_from_lens(normal, lens_dir);
    }

    Vec3 scatter_flux_density(Vec3 normal, Vec3 lens_dir, Vec3 light_dir) {
        return (1.0f / pi) * albedo;
    }

    float scatter_probability_density(Vec3 normal, Vec3 lens_dir, Vec3 light_dir) {
        return 1.0f / pi;
    }

    void sample_scatter_from_lens(Random& random, Vec3 normal, Vec3 lens_dir, Vec3& light_dir, Vec3& result_sample) {
        random.lambert(normal, light_dir);
        result_sample = albedo;
    }

    void sample_scatter_from_light(Random& random, Vec3 normal, Vec3 light_dir, Vec3& lens_dir, Vec3& result_sample) {
        sample_scatter_from_lens(random, normal, light_dir, lens_dir, result_sample);
    }
};

static float geometric(Vec3 na, Vec3 nb, Vec3 d) {
    float dsqr = dotsqr(d);
    return - dot(na, d) * dot(nb, d) / (dsqr * dsqr);
}

struct MainCamera {
    size_t zone_id;
    Vec3 origin;
    Vec3 forward;
    Vec3 right;
    Vec3 down;
    float utan;
    float vtan;

    static MainCamera targeted(World const& world, Vec3 origin, Vec3 target, float ctan) {
        MainCamera camera;
        camera.zone_id = world.zone_index_at(origin);
        camera.origin = origin;
        camera.forward = norm(target - origin);
        Vec3 up{0, 0, 1};
        camera.right = norm(cross(camera.forward, up));
        camera.down = norm(cross(camera.forward, camera.right));
        float aspect = (float)width / (float)height;
        camera.utan = ctan * sqrtf(aspect);
        camera.vtan = ctan / sqrtf(aspect);
        return camera;
    }

    float importance_flux_density(Vec3 direction) const {
        Vec3 dn = norm(direction);
        float cos_at_lens = dot(dn, forward);
        float cos_at_lens_sqr = cos_at_lens * cos_at_lens;
        float cos_at_lens_4 = cos_at_lens_sqr * cos_at_lens_sqr;
        return 1.0f / (4.0f * utan * vtan * cos_at_lens_4);
    }
};

Material test_material(World const& world, size_t zone_id, size_t triangle_index, Vec3& normal) {
    if (triangle_index != World::invalid_index) {
        World::Triangle const& tri = world.zone_trees[zone_id].triangles[triangle_index];
        World::Surface const& surface = world.surfaces[tri.surface_index];
        std::string const& mat_name = world.material_names[surface.material_index];
        normal = norm(world.normals[tri.normal_index]);
        if (surface.flags & World::flag_invisible) {
            //does_transmit = true;
            //flux_gain = Vec3{1.0f, 1.0f, 1.0f};
            //target_zone_id = tri.other_zone_index;
            return Material{};
        //} else if (surface.flags & World::flag_masked) {
        //    does_transmit = true;
        //    flux_gain = Vec3{1.00f, 0.90f, 0.70f};
        //    transmit_offset = 0.001f;
        } else {
            if (false) {
                int i = triangle_index;
                float x = (i % 100) / 100.0f;
                float y = ((i / 100) % 100) / 100.0f;
                float z = ((i / 10000) % 100) / 100.0f;
                return Material::emissive(x, y, z);
            } else {
                if (mat_name == "Fire.FireTexture'XFX.xeolighting'") {
                    return Material::emissive(0.35f, 0.40f, 0.45f);
                } else if (mat_name == "Fire.FireTexture'UTtech1.Misc.donfire'") {
                    return Material::emissive(6.00f, 5.00f, 1.50f);
                } else if (mat_name == "Engine.Texture'ShaneChurch.Lampon5'") {
                    return Material::emissive(10.00f, 3.00f, 5.00f);
                    //} else if (mat_name == "Fire.WetTexture'LavaFX.Lava3'") {
                    //    return Material::emissive(1.00f, 0.30f, 0.10f);
                } else if (mat_name == "Engine.Texture'DecayedS.Ceiling.T-CELING2'") {
                    return Material::emissive(10.00f, 11.00f, 12.00f);
                } else {
                    if (fabsf(normal.z) <= 0.2f) {
                        return Material::diffuse(0.60f, 0.03f, 0.03f);
                    } else {
                        return Material::diffuse(0.30f, 0.30f, 0.40f);
                    }
                }
            }
        }
    } else {
        //emit = Vec3{100.00f, 0.00f, 100.00f};
        normal = Vec3{0.0f, 0.0f, 1.0f};
        return Material{};
    }
}

struct ForwardRay {
    int x, y;
    Vec3 flux;
    Vec3 emission;
    int level;

    static ForwardRay make(int x, int y) {
        return ForwardRay{
            x, y,
            Vec3{1.0f, 1.0f, 1.0f},
            Vec3{0.0f, 0.0f, 0.0f},
            -1,
        };
    }
};

struct PathNode {
    uint64_t prev_node_index;
    int node_level;
    size_t zone_id;
    size_t triangle_index;
    size_t image_pixel;
    Vec3 incoming_sample_density;
    Vec3 incoming_direction;
    Vec3 position;
    Vec3 scattered_sample_density;
};

struct Connector {
    uint64_t lens_path_node;
    uint64_t light_path_node;
};

enum class RayType {
    Forward,
    LightPath,
    Connector,
};

static uint64_t make_ray_id(RayType type, uint8_t epoch, uint64_t data_index) {
    return ((uint64_t)type << 56) | ((uint64_t)epoch << 48) | data_index;
}

static void parse_ray_id(uint64_t ray_id, RayType& type, uint8_t& epoch, uint64_t& data_index) {
    type = (RayType)(ray_id >> 56);
    epoch = (ray_id >> 48);
    data_index = ray_id & 0xffffffffffff;
}

struct Epoch {
    Arena<ForwardRay> fray_arena;
    Arena<PathNode> path_node_arena;
    Arena<Connector> connector_arena;
    std::atomic<uint64_t> use_count = 0;
    std::atomic<intptr_t> image_row_index = 0;

    void reset() {
        fray_arena.clear();
        path_node_arena.clear();
        connector_arena.clear();
        image_row_index.store(0, std::memory_order_relaxed);
    }
};

template<typename T>
class AtomicCounterGuard {
private:
    std::atomic<T>* _counter_ptr;
    T _delta;

public:
    AtomicCounterGuard() {
        _counter_ptr = nullptr;
        _delta = 0;
    }

    AtomicCounterGuard(std::atomic<T>& counter, std::memory_order increment_order = std::memory_order_acquire, T delta = 1) {
        _counter_ptr = &counter;
        _delta = delta;
        if (delta != 0) {
            counter.fetch_add(delta, increment_order);
        }
    }

    AtomicCounterGuard(AtomicCounterGuard&& other) {
        _counter_ptr = other._counter_ptr;
        _delta = other._delta;
        other._delta = 0;
    }

    AtomicCounterGuard& operator=(AtomicCounterGuard&& other) {
        release();
        _counter_ptr = other._counter_ptr;
        _delta = other._delta;
        other._delta = 0;
        return *this;
    }

    ~AtomicCounterGuard() {
        release();
    }

    void release(std::memory_order decrement_order = std::memory_order_release) {
        if (_delta != 0) {
            _counter_ptr->fetch_sub(_delta, decrement_order);
            _delta = 0;
        }
    }

    explicit operator bool() const {
        return _delta != 0;
    }
};

struct Context {
    static constexpr uint8_t epoch_count = 4;

    std::atomic<bool>& pauserequested;
    World const& world;
    RayPipeline& pipeline;
    MainCamera const& camera;
    std::mutex epoch_mutex;
    uint8_t current_epoch_index = 0;
    Epoch epochs[epoch_count];
    std::atomic<uint64_t> ray_counter;
    std::unique_ptr<std::atomic<uint64_t>[]> accum_buffer;
    std::atomic<uint64_t> global_denominator_additive;

    std::atomic<uint64_t>* accum_ptr(intptr_t ix, intptr_t iy) {
        return accum_buffer.get() + 4 * width * iy + 4 * ix;
    }

    void increment_accum(intptr_t ix, intptr_t iy, Vec3 value) {
        uint64_t vr = value.x * accum_num_scale + 0.5f;
        uint64_t vg = value.y * accum_num_scale + 0.5f;
        uint64_t vb = value.z * accum_num_scale + 0.5f;
        std::atomic<uint64_t>* ptr = accum_ptr(ix, iy);
        ptr[0].fetch_add(vr, std::memory_order_relaxed);
        ptr[1].fetch_add(vg, std::memory_order_relaxed);
        ptr[2].fetch_add(vb, std::memory_order_relaxed);
    }

    void increment_accum(intptr_t ix, intptr_t iy, Vec3 value, float weight) {
        increment_accum(ix, iy, value);
        uint64_t vw = weight * accum_den_scale + 0.5f;
        std::atomic<uint64_t>* ptr = accum_ptr(ix, iy);
        ptr[3].fetch_add(vw, std::memory_order_relaxed);
    }

    void increment_weight_global(float weight) {
        uint64_t vw = weight * accum_den_scale + 0.5f;
        global_denominator_additive.fetch_add(vw, std::memory_order_relaxed);
    }
};

struct RetiredRaysProcessor2: Processor {
    Context& context;
    Random random;

    RetiredRaysProcessor2(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
        : Processor(std::move(control_ptr))
        , context(context)
    {
    }

    virtual bool has_pending_work_impl() override {
        return !context.pipeline.completed_packets_empty();
    }

    virtual bool work_impl() override {
        bool work_performed = false;
        RayPipeline::Inserter inserter = context.pipeline.inserter();
        Arena<PathNode>::View path_node_arena_views[Context::epoch_count];
        Arena<Connector>::View connector_arena_views[Context::epoch_count];
        for (size_t epoch = 0; epoch < Context::epoch_count; ++epoch) {
            path_node_arena_views[epoch] = context.epochs[epoch].path_node_arena.view();
            connector_arena_views[epoch] = context.epochs[epoch].connector_arena.view();
        }
        AtomicCounterGuard<uint64_t> epoch_guards[Context::epoch_count];
        Arena<PathNode>::Allocator path_node_allocators[Context::epoch_count];
        Arena<Connector>::Allocator connector_allocators[Context::epoch_count];
        while (true) {
            std::unique_ptr<RayPipeline::RayPacket> packet_ptr = context.pipeline.pop_completed_packet();
            if (!packet_ptr) {
                break;
            }
            work_performed = true;
            for (int i = 0; i < packet_ptr->ray_count; ++i) {
                RayPipeline::RayData ray_data = packet_ptr->extract_ray_data(i);
                RayType ray_type;
                uint8_t epoch;
                uint64_t data_index;
                parse_ray_id(ray_data.extra_data, ray_type, epoch, data_index);
                switch (ray_type) {
                case RayType::LightPath:
                {
                    if (ray_data.hit_world_triangle_index != World::invalid_index) {
                        if (!epoch_guards[epoch]) {
                            epoch_guards[epoch] = AtomicCounterGuard<uint64_t>(context.epochs[epoch].use_count);
                            path_node_allocators[epoch] = context.epochs[epoch].path_node_arena.allocator();
                            connector_allocators[epoch] = context.epochs[epoch].connector_arena.allocator();
                        }
                        PathNode& prev_path_node = path_node_arena_views[epoch][data_index];
                        Vec3 prev_normal;
                        test_material(context.world, prev_path_node.zone_id, prev_path_node.triangle_index, prev_normal);
                        Vec3 hit_normal;
                        Material hit_material = test_material(context.world, packet_ptr->zone_id, ray_data.hit_world_triangle_index, hit_normal);
                        uint64_t next_path_node_index;
                        PathNode& next_path_node = path_node_allocators[epoch].emplace(next_path_node_index);
                        next_path_node.prev_node_index = data_index;
                        next_path_node.node_level = prev_path_node.node_level + 1;
                        next_path_node.zone_id = packet_ptr->zone_id;
                        next_path_node.image_pixel = prev_path_node.image_pixel;
                        next_path_node.triangle_index = ray_data.hit_world_triangle_index;
                        next_path_node.incoming_sample_density = prev_path_node.scattered_sample_density;
                        next_path_node.incoming_direction = -norm(ray_data.direction);
                        next_path_node.position = ray_data.origin + ray_data.max_param * ray_data.direction;
                        {
                            Vec3 direction_to_camera = context.camera.origin - next_path_node.position;
                            if (dot(direction_to_camera, context.camera.forward) < 0 && dot(direction_to_camera, hit_normal) > 0) {
                                uint64_t connector_index;
                                Connector& conn = connector_allocators[epoch].emplace(connector_index);
                                conn.lens_path_node = World::invalid_index;
                                conn.light_path_node = next_path_node_index;
                                inserter.schedule(
                                    packet_ptr->zone_id,
                                    next_path_node.position,
                                    direction_to_camera,
                                    make_ray_id(RayType::Connector, epoch, connector_index),
                                    0.0f, 1.0f);
                                context.epochs[epoch].use_count.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                        //bool does_transmit = false;
                        //Vec3 flux_gain = Vec3{0.0f, 0.0f, 0.0f};
                        float transmit_offset = 0.0f;
                        //Vec3 emit = Vec3{0.0f, 0.0f, 0.0f};
                        size_t target_zone_id = packet_ptr->zone_id;
                        Vec3 light_dir = next_path_node.incoming_direction;
                        float scatter_event_probability = hit_material.scatter_event_probability_from_light(hit_normal, light_dir);
                        bool is_scattered;
                        random.p(scatter_event_probability, is_scattered);
                        if (is_scattered) {
                            Vec3 lens_dir;
                            Vec3 scatter_sample;
                            hit_material.sample_scatter_from_light(random, hit_normal, light_dir, lens_dir, scatter_sample);
                            next_path_node.scattered_sample_density = (1.0f / scatter_event_probability) * elementwise_product(next_path_node.incoming_sample_density, scatter_sample);
                            Vec3 new_origin = ray_data.origin + (ray_data.max_param + transmit_offset) * ray_data.direction;
                            inserter.schedule(
                                target_zone_id,
                                new_origin,
                                lens_dir,
                                make_ray_id(RayType::LightPath, epoch, next_path_node_index));
                        } else {
                            context.epochs[epoch].use_count.fetch_sub(1, std::memory_order_release);
                        }
                    } else {
                        context.epochs[epoch].use_count.fetch_sub(1, std::memory_order_release);
                    }
                    context.ray_counter.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                case RayType::Connector:
                {
                    if (ray_data.max_param == 1.0f) {
                        float cu = dot(ray_data.direction, context.camera.right);
                        float cv = dot(ray_data.direction, context.camera.down);
                        float cw = dot(ray_data.direction, context.camera.forward);
                        float cx = (cu / cw) / context.camera.utan;
                        float cy = (cv / cw) / context.camera.vtan;
                        float sx = 0.5f * width * (1.0f + cx);
                        float sy = 0.5f * height * (1.0f - cy);
                        float dx, dy;
                        random.triangle(dx);
                        random.triangle(dy);
                        intptr_t ix = sx + 1.5f * dx;
                        intptr_t iy = sy + 1.5f * dy;
                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                            Connector& conn = connector_arena_views[epoch][data_index];
                            PathNode& light_path_node = path_node_arena_views[epoch][conn.light_path_node];
                            //if (light_path_node.node_level == 2) {
                                Vec3 dn = norm(ray_data.direction);
                                Vec3 light_normal;
                                Material light_material = test_material(context.world, light_path_node.zone_id, light_path_node.triangle_index, light_normal);
                                Vec3 light_flux_density;
                                if (light_path_node.prev_node_index == World::invalid_index) {
                                    light_flux_density = light_material.emission_flux_density(light_normal, dn);
                                } else {
                                    light_flux_density = light_material.scatter_flux_density(light_normal, dn, light_path_node.incoming_direction);
                                }
                                Vec3 lens_normal;
                                float lens_flux_density;
                                if (conn.lens_path_node == World::invalid_index) {
                                    lens_normal = context.camera.forward;
                                    lens_flux_density = context.camera.importance_flux_density(-dn);
                                } else {
                                    throw "nyi";
                                }
                                float geom = geometric(lens_normal, light_normal, ray_data.direction);
                                assign_elementwise_product(light_flux_density, light_path_node.incoming_sample_density);
                                Vec3 value = geom * lens_flux_density * light_flux_density;
                                context.increment_accum(ix, iy, value);
                            //}
                        }
                    }
                    context.epochs[epoch].use_count.fetch_sub(1, std::memory_order_release);
                    context.ray_counter.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                default:
                    throw std::runtime_error("invalid ray type");
                }
            }
            context.pipeline.collect_ray_packet_spare(std::move(packet_ptr));
        }
        return work_performed;
    }
};

struct NewRaysProcessor2: Processor {
    Context& context;
    Random random;
    size_t packet_count_threshold;

    struct LightTriangle {
        size_t zone_id;
        size_t triangle_index;
        float probability_density;
    };

    DiscreteDistribution light_triangles_distribution;
    std::vector<LightTriangle> light_triangles;

    NewRaysProcessor2(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
        : Processor(std::move(control_ptr))
        , context(context)
    {
        packet_count_threshold = 100000 / context.pipeline.get_params().ray_buffer_size;
        World::Tree const& tree = context.world.zone_trees[context.camera.zone_id];
        std::vector<float> lt_weights;
        float total_weight = 0.0f;
        for (size_t i = 0; i < tree.triangles.size(); ++i) {
            World::Triangle const& tri = tree.triangles[i];
            Vec3 normal;
            Material material = test_material(context.world, context.camera.zone_id, i, normal);
            float density = material.emission.x + material.emission.y + material.emission.z;
            if (density > 0.0f) {
                Vec3 a = context.world.vertices[tri.vertex_indexes[0]];
                Vec3 b = context.world.vertices[tri.vertex_indexes[1]];
                Vec3 c = context.world.vertices[tri.vertex_indexes[2]];
                float area = 0.5f * length(cross(b - a, c - a));
                if (area > 0.0f) {
                    float weight = density * area;
                    light_triangles.push_back(LightTriangle{context.camera.zone_id, i, density});
                    lt_weights.push_back(weight);
                    total_weight += weight;
                }
            }
        }
        light_triangles_distribution = DiscreteDistribution::make(lt_weights);
        for (LightTriangle& lt : light_triangles) {
            lt.probability_density /= total_weight;
        }
    }

    virtual bool has_pending_work_impl() override {
        return context.pipeline.get_total_packet_count() < packet_count_threshold;
    }

    virtual bool work_impl() override {
        bool work_performed = false;
        RayPipeline::Inserter inserter = context.pipeline.inserter();
        uint8_t current_epoch;
        AtomicCounterGuard<uint64_t> current_epoch_guard;
        Arena<PathNode>::Allocator path_node_allocator;
        Arena<Connector>::Allocator connector_allocator;
        while (true) {
            if (context.pipeline.get_total_packet_count() >= packet_count_threshold) {
                break;
            }
            if (context.pauserequested.load(std::memory_order_relaxed)) {
                break;
            }
            if (light_triangles.size() == 0) {
                break;
            }
            if (!current_epoch_guard) {
                std::lock_guard<std::mutex> guard(context.epoch_mutex);
                current_epoch = context.current_epoch_index;
                current_epoch_guard = AtomicCounterGuard<uint64_t>(context.epochs[current_epoch].use_count);
                path_node_allocator = context.epochs[current_epoch].path_node_arena.allocator();
                connector_allocator = context.epochs[current_epoch].connector_arena.allocator();
            }
            intptr_t iy = context.epochs[current_epoch].image_row_index.fetch_add(1, std::memory_order_relaxed);
            if (iy >= height) {
                context.epochs[current_epoch].image_row_index.store(height, std::memory_order_relaxed);
                if (context.pauserequested.load(std::memory_order_relaxed)) {
                    break;
                }
                {
                    std::lock_guard<std::mutex> guard(context.epoch_mutex);
                    current_epoch_guard.release();
                    if (current_epoch == context.current_epoch_index) {
                        uint8_t next_epoch = (current_epoch + 1) % Context::epoch_count;
                        if (context.epochs[next_epoch].use_count.load(std::memory_order_acquire) == 0) {
                            work_performed = true;
                            context.epochs[next_epoch].reset();
                            context.current_epoch_index = next_epoch;
                        } else {
                            break;
                        }
                    }
                }
                continue;
            }
            work_performed = true;
            for (size_t i = 0; i < context.pipeline.get_params().ray_buffer_size; ++i) {
                size_t light_triangle_index;
                light_triangles_distribution.sample(random, light_triangle_index);
                LightTriangle const& lt = light_triangles[light_triangle_index];
                World::Triangle const& wtri = context.world.zone_trees[lt.zone_id].triangles[lt.triangle_index];
                Vec3 normal;
                Material material = test_material(context.world, lt.zone_id, lt.triangle_index, normal);
                Vec3 a = context.world.vertices[wtri.vertex_indexes[0]];
                Vec3 b = context.world.vertices[wtri.vertex_indexes[1]];
                Vec3 c = context.world.vertices[wtri.vertex_indexes[2]];
                float qa, qb, qc;
                random.uniform(qa);
                random.uniform(qb);
                random.uniform(qc);
                float u = fabsf(qa - qb);
                float v = (1.0f - u) * qc;
                Vec3 light_origin = a + u * (b - a) + v * (c - a);
                uint64_t path_node_index;
                PathNode& path_node = path_node_allocator.emplace(path_node_index);
                path_node.prev_node_index = World::invalid_index;
                path_node.node_level = 0;
                path_node.zone_id = lt.zone_id;
                path_node.triangle_index = lt.triangle_index;
                path_node.image_pixel = World::invalid_index;
                path_node.incoming_sample_density = (1.0f / lt.probability_density) * Vec3{1.0f, 1.0f, 1.0f};
                path_node.incoming_direction = normal;
                path_node.position = light_origin;
                {
                    Vec3 light_dir;
                    Vec3 emit_sample;
                    material.sample_emission(random, normal, light_dir, emit_sample);
                    path_node.scattered_sample_density = elementwise_product(path_node.incoming_sample_density, emit_sample);
                    inserter.schedule(
                        lt.zone_id,
                        light_origin,
                        light_dir,
                        make_ray_id(RayType::LightPath, current_epoch, path_node_index));
                    context.epochs[current_epoch].use_count.fetch_add(1, std::memory_order_relaxed);
                }
                Vec3 direction_to_camera = context.camera.origin - light_origin;
                if (dot(direction_to_camera, context.camera.forward) < 0 && dot(direction_to_camera, normal) > 0) {
                    uint64_t connector_index;
                    Connector& conn = connector_allocator.emplace(connector_index);
                    conn.lens_path_node = World::invalid_index;
                    conn.light_path_node = path_node_index;
                    inserter.schedule(
                        lt.zone_id,
                        light_origin,
                        direction_to_camera,
                        make_ray_id(RayType::Connector, current_epoch, connector_index),
                        0.0f, 1.0f);
                    context.epochs[current_epoch].use_count.fetch_add(1, std::memory_order_relaxed);
                }
                context.increment_weight_global(1.0f / (float)(width * height));
            }
        }
        return work_performed;
    }
};

void RendererThread::ThreadFunc()
{
    try {
        RayPipelineParams params;
        //params.triangle_rep = RayPipelineParams::TriangleRep::TriplePointIndexed;
        params.triangle_rep = RayPipelineParams::TriangleRep::TriplePointImmediate;
        //params.triangle_rep = RayPipelineParams::TriangleRep::MiddleCoordPermuted;
        //params.triangle_rep = RayPipelineParams::TriangleRep::BoxCenteredUV;
        params.max_chunk_height = 8;
        params.ray_buffer_size = 0x100;
        World world = load_world("Data\\map.bvh");
        //World world = load_test_world();
        RayPipeline pipeline(world, params);
        //Random random;

        MainCamera camera = MainCamera::targeted(world, Vec3{110, -738, 63}, Vec3{-653, -1641, 18}, 1.0f);
        //MainCamera camera = MainCamera::targeted(
        //    world,
        //    Vec3{108, -1150, 55},
        //    Vec3{64, -1150, 236},
        //    1.0f);

        Context context{pauserequested, world, pipeline, camera};
        context.ray_counter.store(0, std::memory_order_relaxed);
        context.accum_buffer = std::make_unique<std::atomic<uint64_t>[]>(width * height * 4);
        for (size_t i = 0; i < width * height * 4; ++i) {
            context.accum_buffer[i].store(0, std::memory_order_relaxed);
        }
        context.global_denominator_additive.store(0, std::memory_order_relaxed);

        std::shared_ptr<ProcessorControl> this_control_ptr = std::make_shared<ProcessorControl>();
        for (int i = 0; i < 2; ++i) {
            Scheduler::register_processor(std::make_shared<NewRaysProcessor2>(this_control_ptr, context));
        }
        Scheduler::register_processor(std::make_shared<RetiredRaysProcessor2>(this_control_ptr, context));

        uint64_t start_time = GetTickCount64() + 15000;

        while (!rterminate.load(std::memory_order_relaxed)) {
            using namespace std::chrono_literals;

            std::this_thread::sleep_for(100ms);

            {
                auto& bits = bitbuf.Back();
                bits.resize(width * height * 4);
                uint8_t* pixels = bits.data();
                uint64_t global_den = context.global_denominator_additive.load(std::memory_order_relaxed);
                for (intptr_t iy = 0; iy < height; ++iy) {
                    uint8_t* line = pixels + iy * width * 4;
                    for (intptr_t ix = 0; ix < width; ++ix) {
                        uint8_t* pixel = line + ix * 4;
                        std::atomic<uint64_t>* accum_ptr = context.accum_buffer.get() + 4 * width * iy + 4 * ix;
                        float w = global_den + accum_ptr[3].load(std::memory_order_relaxed);
                        float r = 0, g = 0, b = 0;
                        if (w != 0) {
                            float scale = (accum_den_scale / accum_num_scale) / w;
                            r = scale * (float)accum_ptr[0].load(std::memory_order_relaxed);
                            g = scale * (float)accum_ptr[1].load(std::memory_order_relaxed);
                            b = scale * (float)accum_ptr[2].load(std::memory_order_relaxed);
                        }
                        tonemap(r, g, b, pixel);
                    }
                }
                bitbuf.Publish();
                InvalidateRgn(hwnd, nullptr, false);
            }

            char buf[1024];
            if (pauserequested.load(std::memory_order_relaxed)) {
                start_time = GetTickCount64() + 15000;
                snprintf(
                    buf, sizeof(buf),
                    "rt | packets: %llu | flushing",
                    pipeline.get_total_packet_count());
            } else {
                uint64_t counter = context.ray_counter.load(std::memory_order_relaxed);
                uint64_t time = GetTickCount64();
                if (time <= start_time) {
                    context.ray_counter.store(0, std::memory_order_relaxed);
                    snprintf(
                        buf, sizeof(buf),
                        "rt | packets: %llu | warming up (%llus)",
                        pipeline.get_total_packet_count(),
                        (start_time - time) / 1000);
                } else {
                    snprintf(
                        buf, sizeof(buf),
                        "rt | packets: %llu | Krays per second: %llu",
                        pipeline.get_total_packet_count(),
                        counter / (time - start_time + 1));
                }
            }
            SendMessageTimeoutA(
                hwnd, WM_SETTEXT, (WPARAM)nullptr, (LPARAM)buf,
                SMTO_NORMAL, 100, nullptr);
        }

        this_control_ptr->set_dead();

        //std::vector<uint8_t> pixel_buffer(width * height * 4, 255);
        //uint8_t* pixels = pixel_buffer.data();
        //for (intptr_t iy = 0; iy < height; ++iy) {
        //    uint8_t* line = pixels + iy * width * 4;
        //    for (intptr_t ix = 0; ix < width; ++ix) {
        //        uint8_t* pixel = line + ix * 4;
        //        float cx = 2.0f * (ix + 0.5f) / width - 1.0f;
        //        float cy = 1.0f - 2.0f * (iy + 0.5f) / height;
        //        float cu = cx * utan;
        //        float cv = cy * vtan;
        //        Ray rd(zone_index, origin, forward + cu * right + cv * down);
        //        rp.trace_immediately(rd);
        //        int index = 0;
        //        if (rd.hit_triangle_index() != World::invalid_index) {
        //            index = world.zone_trees[zone_index].triangles[rd.hit_triangle_index()].surface_index;
        //        }
        //        pixel[0] = (index & 0xf) * 0x11;
        //        pixel[1] = ((index >> 4) & 0xf) * 0x11;
        //        pixel[2] = ((index >> 8) & 0xf) * 0x11;
        //        pixel[3] = 255;
        //        //float x = rd.max_param() / 1000;
        //        //float x = rd.hit_triangle_pos.x + rd.hit_triangle_pos.y + rd.hit_triangle_pos.z;
        //        //Tonemap(FDisp(x, x / 10, x / 100), pixel);
        //        //Tonemap(FDisp(rd.hit_triangle_pos().x, rd.hit_triangle_pos().y, rd.hit_triangle_pos().z), pixel);
        //    }
        //    {
        //        auto& bits = bitbuf.Back();
        //        bits.resize(width * height * 4);
        //        memcpy(bits.data(), pixels, pixel_buffer.size());
        //        bitbuf.Publish();
        //        InvalidateRgn(hwnd, nullptr, false);
        //    }
        //    if (rterminate.load(std::memory_order_relaxed)) {
        //        return;
        //    }
        //}

        //std::unique_ptr<SamplerBase> pintegrator{ new BidirSampler(width, height) };
        //std::unique_ptr<SamplerBase> pintegrator{ new ForwardSampler(width, height) };
        //std::unique_ptr<SamplerBase> pintegrator{ new ParameterSampler(width, height) };
        //int64_t time = GetTickCount64();
        //int iterindex = 0;
        //while (!rterminate.load(std::memory_order_relaxed)) {
        //    if (exportrequested.load(std::memory_order_relaxed)) {
        //        pintegrator->Export();
        //        exportrequested.store(false, std::memory_order_relaxed);
        //    }
        //    if (pauserequested.load(std::memory_order_relaxed)) {
        //        std::this_thread::yield(); 
        //    } else {
        //        iterindex += 1;
        //        pintegrator->Iterate();
        //        auto& bits = bitbuf.Back();
        //        bits.resize(width * height * 4);
        //        uint8_t* pixels = bits.data();
        //        for (intptr_t iy = 0; iy < height; ++iy) {
        //            uint8_t* line = pixels + iy * width * 4;
        //            for (intptr_t ix = 0; ix < width; ++ix) {
        //                uint8_t* pixel = line + ix * 4;
        //                FDisp value = exposure * pintegrator->GetValue(ix, iy);
        //                Tonemap(value, pixel);
        //            }
        //        }
        //        auto& perf = pintegrator->GetPerfInfo();
        //        char buf[1024];
        //        if (IsSavedIndex(iterindex)) {
        //            png_image pi = {};
        //            pi.opaque = nullptr;
        //            pi.version = PNG_IMAGE_VERSION;
        //            pi.width = width;
        //            pi.height = height;
        //            pi.format = PNG_FORMAT_BGRA;
        //            pi.flags = 0;
        //            pi.colormap_entries = 0;
        //            snprintf(
        //                buf, sizeof(buf),
        //                "D:\\rt\\output\\%.5d.png",
        //                iterindex);
        //            png_image_write_to_file(&pi, buf, false, bits.data(), -width * 4, nullptr);
        //        }
        //        bitbuf.Publish();
        //        InvalidateRgn(hwnd, nullptr, false);
        //        snprintf(
        //            buf, sizeof(buf),
        //            "rt | %i | %lli | %8.6f",
        //            iterindex, GetTickCount64() - time, perf.error);
        //        SendMessageTimeoutA(
        //            hwnd, WM_SETTEXT, (WPARAM)nullptr, (LPARAM)buf,
        //            SMTO_NORMAL, 100, nullptr);
        //        if (IsImportantIndex(iterindex) && !pauserequested.load(std::memory_order_relaxed)) {
        //            SetForegroundWindow(hwnd);
        //            pauserequested.store(true, std::memory_order_relaxed);
        //        }
        //    }
        //}
    } catch (std::exception const& e) {
        MessageBoxA(hwnd, e.what(), "error", 0);
        SendMessageTimeoutA(
            hwnd, WM_CLOSE, (WPARAM)nullptr, (LPARAM)nullptr,
            SMTO_NORMAL, 100, nullptr);
    }
}

RendererThread::RendererThread(HWND hwnd)
    : hwnd(hwnd)
{
    rterminate.store(false, std::memory_order_relaxed);
    exportrequested.store(false, std::memory_order_relaxed);
    pauserequested.store(false, std::memory_order_relaxed);
    rthread = std::thread{ &RendererThread::ThreadFunc, this };
}

RendererThread::~RendererThread()
{
    rterminate.store(true, std::memory_order_relaxed);
    rthread.join();
}

void RendererThread::GetWindowSize(int& w, int& h)
{
    w = width;
    h = height;
}

void RendererThread::DrawFrame(HDC dc)
{
    auto& bits = bitbuf.Forward();
    if (bits.empty())
        return;
    BITMAPINFOHEADER bih;
    bih.biSize = sizeof(bih);
    bih.biWidth = width;
    bih.biHeight = height;
    bih.biPlanes = 1;
    bih.biBitCount = 32;
    bih.biCompression = BI_RGB;
    bih.biSizeImage = 0;
    bih.biXPelsPerMeter = 1;
    bih.biYPelsPerMeter = 1;
    bih.biClrUsed = 0;
    bih.biClrImportant = 0;
    SetDIBitsToDevice(
        dc,
        0, 0,
        width, height,
        0, 0,
        0, height,
        bits.data(),
        (BITMAPINFO*)&bih,
        DIB_RGB_COLORS);
}

void RendererThread::Export()
{
    exportrequested.store(true, std::memory_order_relaxed);
}

void RendererThread::SetPause(bool value)
{
    pauserequested.store(value, std::memory_order_relaxed);
}
