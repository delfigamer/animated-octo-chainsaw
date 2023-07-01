#include "RendererThread.h"
#include "World.h"
#include "RayPipeline.h"
#include "Threading.h"
//#include <png.h>

constexpr int width = 1600;
constexpr int height = 900;
constexpr int lines_per_epoch = 100;
constexpr int epochs_per_frame = (height + lines_per_epoch - 1) / lines_per_epoch;
constexpr int light_samples_per_epoch = lines_per_epoch * 1600;
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

struct Surface {
    Vec3 normal;
    Vec3 albedo = Vec3{0, 0, 0};
    Vec3 emission = Vec3{0, 0, 0};

    static Surface black(Vec3 normal) {
        Surface m;
        m.normal = normal;
        return m;
    }

    static Surface emissive(Vec3 normal, float ex, float ey, float ez) {
        Surface m;
        m.normal = normal;
        m.emission = Vec3{ex, ey, ez};
        return m;
    }

    static Surface diffuse(Vec3 normal, float ax, float ay, float az) {
        Surface m;
        m.normal = normal;
        m.albedo = Vec3{ax, ay, az};
        return m;
    }

    Vec3 emission_flux_density(Vec3 lens_dir) const {
        return emission;
    }

    float emission_probability_density(Vec3 lens_dir) const {
        return 1.0f / pi;
    }

    void sample_emission(Random& random, Vec3& light_dir, Vec3& result_value) const {
        random.lambert(normal, light_dir);
        result_value = pi * emission;
    }

    float scatter_event_probability_from_lens(Vec3 lens_dir) const {
        return fmaxf(fmaxf(albedo.x, albedo.y), albedo.z);
    }

    float scatter_event_probability_from_light(Vec3 lens_dir) const {
        return scatter_event_probability_from_lens(lens_dir);
    }

    Vec3 scatter_flux_density(Vec3 lens_dir, Vec3 light_dir) const {
        return (1.0f / pi) * albedo;
    }

    float scatter_probability_density(Vec3 lens_dir, Vec3 light_dir) const {
        return 1.0f / pi;
    }

    void sample_scatter_from_lens(Random& random, Vec3 lens_dir, Vec3& light_dir, Vec3& result_value) const {
        random.lambert(normal, light_dir);
        result_value = albedo;
    }

    void sample_scatter_from_light(Random& random, Vec3 light_dir, Vec3& lens_dir, Vec3& result_value) const {
        sample_scatter_from_lens(random, light_dir, lens_dir, result_value);
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

    void pixel_to_world(Random& random, int ix, int iy, Vec3& direction) const {
        float dx, dy;
        random.triangle(dx);
        random.triangle(dy);
        float cnx = (2.0f / (float)width)  * ((float)ix + 1.5f * dx + 0.5f);
        float cny = (2.0f / (float)height) * ((float)iy + 1.5f * dy + 0.5f);
        float cu = (cnx - 1.0f) * utan;
        float cv = (1.0f - cny) * vtan;
        direction = forward + cu * right + cv * down;
    }

    bool world_to_pixel(Random& random, Vec3 direction, int& ix, int& iy) const {
        float cw = dot(direction, forward);
        if (cw < 0) {
            float cu = dot(direction, right);
            float cv = dot(direction, down);
            float cx = (cu / cw) / utan;
            float cy = (cv / cw) / vtan;
            float sx = 0.5f * width * (1.0f + cx);
            float sy = 0.5f * height * (1.0f - cy);
            float dx, dy;
            random.triangle(dx);
            random.triangle(dy);
            ix = sx + 1.5f * dx;
            iy = sy + 1.5f * dy;
            return ix >= 0 && ix < width && iy >= 0 && iy < height;
        } else {
            return false;
        }
    }
};

Surface test_material_surface(World const& world, size_t zone_id, size_t triangle_index) {
    if (triangle_index != World::invalid_index) {
        World::Triangle const& tri = world.zone_trees[zone_id].triangles[triangle_index];
        World::Surface const& surface = world.surfaces[tri.surface_index];
        std::string const& mat_name = world.material_names[surface.material_index];
        Vec3 normal = norm(world.normals[tri.normal_index]);
        if (surface.flags & World::flag_invisible) {
            //does_transmit = true;
            //flux_gain = Vec3{1.0f, 1.0f, 1.0f};
            //target_zone_id = tri.other_zone_index;
            return Surface::black(normal);
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
                return Surface::emissive(normal, x, y, z);
            } else {
                if (mat_name == "Fire.FireTexture'XFX.xeolighting'") {
                    return Surface::emissive(normal, 0.35f, 0.40f, 0.45f);
                } else if (mat_name == "Fire.FireTexture'UTtech1.Misc.donfire'") {
                    return Surface::emissive(normal, 6.00f, 5.00f, 1.50f);
                } else if (mat_name == "Engine.Texture'ShaneChurch.Lampon5'") {
                    return Surface::emissive(normal, 10.00f, 3.00f, 5.00f);
                //} else if (mat_name == "Fire.WetTexture'LavaFX.Lava3'") {
                //    return Surface::emissive(normal, 1.00f, 0.30f, 0.10f);
                } else if (mat_name == "Engine.Texture'DecayedS.Ceiling.T-CELING2'") {
                    return Surface::emissive(normal, 10.00f, 11.00f, 12.00f);
                } else {
                    if (fabsf(normal.z) <= 0.2f) {
                        return Surface::diffuse(normal, 0.60f, 0.03f, 0.03f);
                    } else {
                        return Surface::diffuse(normal, 0.30f, 0.30f, 0.40f);
                    }
                }
            }
        }
    } else {
        //emit = Vec3{100.00f, 0.00f, 100.00f};
        return Surface::black(Vec3{0.0f, 0.0f, 1.0f});
    }
}

struct PathNode {
    uint64_t prev_node_index;
    int node_level;
    int path_index;
    Vec3 incoming_value_density;
    Vec3 incoming_direction;
    Vec3 position;
    Surface surface;
    Vec3 scattered_value_density;
};

struct Connector {
    uint64_t lens_path_node;
    uint64_t light_path_node;
    int ix;
    int iy;
    Vec3 value;
};

enum class RayType {
    LensPath,
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

constexpr uint64_t path_origin_bit = (uint64_t)1 << 63;

struct Epoch {
    Arena<PathNode> path_node_arena;
    Arena<Connector> connector_arena;
    std::atomic<uint64_t> use_count = 0;
    std::atomic<int> loader_progress = 0;
    std::atomic<int> connector_progress = 0;
    std::atomic<bool> completion_recorded = false;
    size_t epoch_number = 0;

    void reset(size_t new_epoch_number) {
        path_node_arena.clear();
        connector_arena.clear();
        loader_progress.store(0, std::memory_order_relaxed);
        connector_progress.store(0, std::memory_order_relaxed);
        completion_recorded.store(false, std::memory_order_relaxed);
        epoch_number = new_epoch_number;
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
    std::atomic<bool>& steprequested;
    World const& world;
    RayPipeline& pipeline;
    MainCamera const& camera;
    std::mutex epoch_mutex;
    uint8_t current_epoch_index = 0;
    Epoch epochs[epoch_count];
    std::atomic<size_t> total_frames_completed;
    std::atomic<uint64_t> ray_counter;
    std::unique_ptr<std::atomic<uint64_t>[]> accum_buffer;
    std::atomic<uint64_t> global_denominator_additive;

    bool is_paused() {
        return pauserequested.load(std::memory_order_relaxed) && !steprequested.load(std::memory_order_relaxed);
    }

    void consume_step_request() {
        steprequested.store(false, std::memory_order_relaxed);
    }

    std::atomic<uint64_t>* accum_ptr(int ix, int iy) {
        return accum_buffer.get() + 4 * width * iy + 4 * ix;
    }

    Vec3 color_at(int ix, int iy) {
        std::atomic<uint64_t>* accum_ptr = accum_buffer.get() + 4 * width * iy + 4 * ix;
        float w = global_denominator_additive.load(std::memory_order_relaxed) + accum_ptr[3].load(std::memory_order_relaxed);
        Vec3 result{0, 0, 0};
        if (w != 0) {
            float scale = (accum_den_scale / accum_num_scale) / w;
            result.x = scale * (float)accum_ptr[0].load(std::memory_order_relaxed);
            result.y = scale * (float)accum_ptr[1].load(std::memory_order_relaxed);
            result.z = scale * (float)accum_ptr[2].load(std::memory_order_relaxed);
        }
        return result;
    }

    void increment_accum_color(int ix, int iy, Vec3 value) {
        uint64_t vr = value.x * accum_num_scale + 0.5f;
        uint64_t vg = value.y * accum_num_scale + 0.5f;
        uint64_t vb = value.z * accum_num_scale + 0.5f;
        std::atomic<uint64_t>* ptr = accum_ptr(ix, iy);
        if (vr != 0) ptr[0].fetch_add(vr, std::memory_order_relaxed);
        if (vg != 0) ptr[1].fetch_add(vg, std::memory_order_relaxed);
        if (vb != 0) ptr[2].fetch_add(vb, std::memory_order_relaxed);
    }

    void increment_accum_weight(int ix, int iy, float weight) {
        uint64_t vw = weight * accum_den_scale + 0.5f;
        std::atomic<uint64_t>* ptr = accum_ptr(ix, iy);
        ptr[3].fetch_add(vw, std::memory_order_relaxed);
    }

    void increment_weight_global(float weight) {
        uint64_t vw = weight * accum_den_scale + 0.5f;
        global_denominator_additive.fetch_add(vw, std::memory_order_relaxed);
    }
};

struct CurrentEpochView {
    Context& context;
    uint8_t current_epoch;
    AtomicCounterGuard<uint64_t> current_epoch_guard;
    Arena<PathNode>::Allocator path_node_allocator;
    Arena<Connector>::Allocator connector_allocator;

    CurrentEpochView(Context& context)
        : context(context)
    {
    }

    explicit operator bool() const {
        return (bool)current_epoch_guard;
    }

    void initialize() {
        if (!current_epoch_guard) {
            std::lock_guard<std::mutex> guard(context.epoch_mutex);
            current_epoch = context.current_epoch_index;
            current_epoch_guard = AtomicCounterGuard<uint64_t>(context.epochs[current_epoch].use_count);
            path_node_allocator = context.epochs[current_epoch].path_node_arena.allocator();
            connector_allocator = context.epochs[current_epoch].connector_arena.allocator();
        }
    }

    bool advance_loader_progress(bool& work_performed, int& iq, int limit) {
        while (true) {
            initialize();
            iq = context.epochs[current_epoch].loader_progress.fetch_add(1, std::memory_order_relaxed);
            if (iq < limit) {
                context.consume_step_request();
                return true;
            }
            context.epochs[current_epoch].loader_progress.store(limit, std::memory_order_relaxed);
            bool is_final_epoch_of_frame = (context.epochs[current_epoch].epoch_number + 1) % epochs_per_frame == 0;
            if (!context.epochs[current_epoch].completion_recorded.exchange(true, std::memory_order_relaxed)) {
                if (is_final_epoch_of_frame) {
                    context.total_frames_completed.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (is_final_epoch_of_frame && context.is_paused()) {
                return false;
            }
            {
                std::lock_guard<std::mutex> guard(context.epoch_mutex);
                current_epoch_guard.release();
                if (current_epoch == context.current_epoch_index) {
                    uint8_t next_epoch = (current_epoch + 1) % Context::epoch_count;
                    if (context.epochs[next_epoch].use_count.load(std::memory_order_acquire) != 0) {
                        return false;
                    }
                    work_performed = true;
                    context.epochs[next_epoch].reset(context.epochs[current_epoch].epoch_number + 1);
                    context.current_epoch_index = next_epoch;
                }
            }
        }
    }

    Epoch* operator->() {
        return &context.epochs[current_epoch];
    }
};

struct EpochView {
    Epoch* epoch_ptr;
    Arena<PathNode>::View path_node_arena_view;
    Arena<Connector>::View connector_arena_view;
    AtomicCounterGuard<uint64_t> epoch_guard;
    Arena<PathNode>::Allocator path_node_allocator;
    Arena<Connector>::Allocator connector_allocator;

    EpochView() {
        epoch_ptr = nullptr;
    }

    EpochView(Epoch& epoch)
        : epoch_ptr(&epoch)
        , path_node_arena_view(epoch.path_node_arena)
        , connector_arena_view(epoch.connector_arena)
    {
    }

    EpochView(EpochView&& other) = default;
    EpochView& operator=(EpochView&& other) = default;

    void initialize_allocators() {
        if (!epoch_guard) {
            epoch_guard = AtomicCounterGuard<uint64_t>(epoch_ptr->use_count);
            path_node_allocator = epoch_ptr->path_node_arena.allocator();
            connector_allocator = epoch_ptr->connector_arena.allocator();
        }
    }

    PathNode& path_node_at(uint64_t index) {
        return path_node_arena_view[index];
    }

    Connector& connector_at(uint64_t index) {
        return connector_arena_view[index];
    }

    template<typename... Ts>
    PathNode& emplace_path_node(uint64_t& index, Ts&&... args) {
        initialize_allocators();
        return path_node_allocator.emplace(index, std::forward<Ts>(args)...);
    }

    template<typename... Ts>
    Connector& emplace_connector(uint64_t& index, Ts&&... args) {
        initialize_allocators();
        return connector_allocator.emplace(index, std::forward<Ts>(args)...);
    }

    Epoch* operator->() {
        return epoch_ptr;
    }
};

static PathNode camera_node(MainCamera const& camera, int ix, int iy) {
    PathNode path_node;
    path_node.prev_node_index = path_origin_bit | (iy * width + ix);
    path_node.node_level = 0;
    path_node.incoming_value_density = Vec3{1.0f, 1.0f, 1.0f};
    path_node.incoming_direction = camera.forward;
    path_node.position = camera.origin;
    path_node.surface = Surface::black(camera.forward);
}

static void get_lens_path_origin_pixel(EpochView& epoch_view, PathNode const& base_path_node, int& ix, int& iy) {
    PathNode const* iter_node_ptr = &base_path_node;
    while (true) {
        if (iter_node_ptr->prev_node_index & path_origin_bit) {
            uint64_t pixel = iter_node_ptr->prev_node_index & ~path_origin_bit;
            ix = pixel % width;
            iy = pixel / width;
            break;
        } else {
            iter_node_ptr = &epoch_view.path_node_at(iter_node_ptr->prev_node_index);
        }
    }
}

static bool get_connector_params(MainCamera const& camera, EpochView& epoch_view, PathNode const& lens_node, PathNode const& light_node, int& ix, int& iy, Vec3 value) {
    Vec3 delta = lens_node.position - light_node.position;
    if (dot(delta, lens_node.surface.normal) < 0 && dot(delta, light_node.surface.normal) > 0) {
        get_lens_path_origin_pixel(epoch_view, lens_node, ix, iy);
        Vec3 dn = norm(delta);
        Vec3 light_flux_density;
        if (light_node.prev_node_index & path_origin_bit) {
            light_flux_density = light_node.surface.scatter_flux_density(dn, light_node.incoming_direction);
        } else {
            light_flux_density = light_node.surface.emission_flux_density(dn);
        }
        Vec3 lens_flux_density;
        if (lens_node.prev_node_index & path_origin_bit) {
            float f = camera.importance_flux_density(-dn);
            lens_flux_density = Vec3{f, f, f};
        } else {
            lens_flux_density = lens_node.surface.scatter_flux_density(lens_node.incoming_direction, -dn);
        }
        float geom = geometric(lens_node.surface.normal, light_node.surface.normal, delta);
        assign_elementwise_product(light_flux_density, light_node.incoming_value_density);
        assign_elementwise_product(lens_flux_density, lens_node.incoming_value_density);
        value = geom * elementwise_product(lens_flux_density, light_flux_density);
        return true;
    } else {
        return false;
    }
}

struct HandlerProcessor: Processor {
    Context& context;
    Random random;

    HandlerProcessor(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
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
        EpochView epoch_views[Context::epoch_count];
        for (size_t epoch = 0; epoch < Context::epoch_count; ++epoch) {
            epoch_views[epoch] = EpochView(context.epochs[epoch]);
        }
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
                case RayType::LensPath:
                {
                    if (ray_data.hit_world_triangle_index != World::invalid_index) {
                        PathNode& prev_path_node = epoch_views[epoch].path_node_at(data_index);
                        uint64_t next_path_node_index;
                        PathNode& next_path_node = epoch_views[epoch].emplace_path_node(next_path_node_index);
                        next_path_node.prev_node_index = data_index;
                        next_path_node.node_level = prev_path_node.node_level + 1;
                        next_path_node.incoming_value_density = prev_path_node.scattered_value_density;
                        next_path_node.incoming_direction = -norm(ray_data.direction);
                        next_path_node.position = ray_data.origin + ray_data.max_param * ray_data.direction;
                        next_path_node.surface = test_material_surface(context.world, packet_ptr->zone_id, ray_data.hit_world_triangle_index);
                        Vec3 direct_emission = next_path_node.surface.emission_flux_density(next_path_node.incoming_direction);
                        if (direct_emission.x != 0.0f || direct_emission.y != 0.0f || direct_emission.z != 0.0f) {
                            int ix, iy;
                            get_lens_path_origin_pixel(epoch_views[epoch], prev_path_node, ix, iy);
                            context.increment_accum_color(ix, iy, elementwise_product(direct_emission, next_path_node.incoming_value_density));
                        }
                        //bool does_transmit = false;
                        //Vec3 flux_gain = Vec3{0.0f, 0.0f, 0.0f};
                        float transmit_offset = 0.0f;
                        //Vec3 emit = Vec3{0.0f, 0.0f, 0.0f};
                        size_t target_zone_id = packet_ptr->zone_id;
                        Vec3 lens_dir = next_path_node.incoming_direction;
                        float scatter_event_probability = next_path_node.surface.scatter_event_probability_from_lens(lens_dir);
                        bool is_scattered;
                        random.p(scatter_event_probability, is_scattered);
                        if (is_scattered) {
                            Vec3 light_dir;
                            Vec3 scatter_value;
                            next_path_node.surface.sample_scatter_from_lens(random, lens_dir, light_dir, scatter_value);
                            next_path_node.scattered_value_density = (1.0f / scatter_event_probability) * elementwise_product(next_path_node.incoming_value_density, scatter_value);
                            Vec3 new_origin = ray_data.origin + (ray_data.max_param + transmit_offset) * ray_data.direction;
                            inserter.schedule(
                                target_zone_id,
                                new_origin,
                                light_dir,
                                make_ray_id(RayType::LensPath, epoch, next_path_node_index));
                        } else {
                            epoch_views[epoch]->use_count.fetch_sub(1, std::memory_order_release);
                        }
                    } else {
                        epoch_views[epoch]->use_count.fetch_sub(1, std::memory_order_release);
                    }
                    context.ray_counter.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                case RayType::LightPath:
                {
                    if (ray_data.hit_world_triangle_index != World::invalid_index) {
                        PathNode& prev_path_node = epoch_views[epoch].path_node_at(data_index);
                        uint64_t next_path_node_index;
                        PathNode& next_path_node = epoch_views[epoch].emplace_path_node(next_path_node_index);
                        next_path_node.prev_node_index = data_index;
                        next_path_node.node_level = prev_path_node.node_level + 1;
                        next_path_node.incoming_value_density = prev_path_node.scattered_value_density;
                        next_path_node.incoming_direction = -norm(ray_data.direction);
                        next_path_node.position = ray_data.origin + ray_data.max_param * ray_data.direction;
                        next_path_node.surface = test_material_surface(context.world, packet_ptr->zone_id, ray_data.hit_world_triangle_index);
                        {
                            Vec3 direction_to_camera = context.camera.origin - next_path_node.position;
                            if (dot(direction_to_camera, next_path_node.surface.normal) > 0) {
                                int ix, iy;
                                if (context.camera.world_to_pixel(random, direction_to_camera, ix, iy)) {
                                    Vec3 dn = norm(direction_to_camera);
                                    Vec3 light_flux_density = next_path_node.surface.scatter_flux_density(dn, next_path_node.incoming_direction);
                                    Vec3 lens_normal = context.camera.forward;
                                    float lens_flux_density = context.camera.importance_flux_density(-dn);
                                    float geom = geometric(lens_normal, next_path_node.surface.normal, direction_to_camera);
                                    assign_elementwise_product(light_flux_density, next_path_node.incoming_value_density);
                                    Vec3 value = geom * lens_flux_density * light_flux_density;
                                    uint64_t connector_index;
                                    Connector& conn = epoch_views[epoch].emplace_connector(connector_index);
                                    conn.lens_path_node = World::invalid_index;
                                    conn.light_path_node = next_path_node_index;
                                    conn.ix = ix;
                                    conn.iy = iy;
                                    conn.value = value;
                                    inserter.schedule(
                                        packet_ptr->zone_id,
                                        next_path_node.position,
                                        direction_to_camera,
                                        make_ray_id(RayType::Connector, epoch, connector_index),
                                        0.0f, 1.0f);
                                    epoch_views[epoch]->use_count.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                        }
                        //bool does_transmit = false;
                        //Vec3 flux_gain = Vec3{0.0f, 0.0f, 0.0f};
                        float transmit_offset = 0.0f;
                        //Vec3 emit = Vec3{0.0f, 0.0f, 0.0f};
                        size_t target_zone_id = packet_ptr->zone_id;
                        Vec3 light_dir = next_path_node.incoming_direction;
                        float scatter_event_probability = next_path_node.surface.scatter_event_probability_from_light(light_dir);
                        bool is_scattered;
                        random.p(scatter_event_probability, is_scattered);
                        if (is_scattered) {
                            Vec3 lens_dir;
                            Vec3 scatter_value;
                            next_path_node.surface.sample_scatter_from_light(random, light_dir, lens_dir, scatter_value);
                            next_path_node.scattered_value_density = (1.0f / scatter_event_probability) * elementwise_product(next_path_node.incoming_value_density, scatter_value);
                            Vec3 new_origin = ray_data.origin + (ray_data.max_param + transmit_offset) * ray_data.direction;
                            inserter.schedule(
                                target_zone_id,
                                new_origin,
                                lens_dir,
                                make_ray_id(RayType::LightPath, epoch, next_path_node_index));
                        } else {
                            epoch_views[epoch]->use_count.fetch_sub(1, std::memory_order_release);
                        }
                    } else {
                        epoch_views[epoch]->use_count.fetch_sub(1, std::memory_order_release);
                    }
                    context.ray_counter.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                case RayType::Connector:
                {
                    if (ray_data.max_param == 1.0f) {
                        Connector& conn = epoch_views[epoch].connector_at(data_index);
                        size_t lens_path_length = 0;
                        if (conn.lens_path_node != World::invalid_index) {
                            lens_path_length = epoch_views[epoch].path_node_at(conn.lens_path_node).node_level;
                        }
                        size_t light_path_length = 0;
                        if (conn.light_path_node != World::invalid_index) {
                            light_path_length = epoch_views[epoch].path_node_at(conn.light_path_node).node_level;
                        }
                        if (lens_path_length == 0 && light_path_length == 1) {
                            context.increment_accum_color(conn.ix, conn.iy, conn.value);
                        }
                    }
                    epoch_views[epoch]->use_count.fetch_sub(1, std::memory_order_release);
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

struct ConnectorProcessor: Processor {
    Context& context;
    Random random;

    ConnectorProcessor(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
        : Processor(std::move(control_ptr))
        , context(context)
    {
    }

    virtual bool has_pending_work_impl() override {
        return true;
    }

    virtual bool work_impl() override {
        bool work_performed = false;
        RayPipeline::Inserter inserter = context.pipeline.inserter();
        CurrentEpochView current_epoch_view(context);
        //while (true) {
        //    if (context.pipeline.get_total_packet_count() >= packet_count_threshold) {
        //        break;
        //    }
        //    if (light_triangles.size() == 0) {
        //        break;
        //    }
        //    int iq;
        //    if (!current_epoch_view.advance_loader_progress(work_performed, iq, lines_per_epoch)) {
        //        break;
        //    }
        //    work_performed = true;
        //    int light_i_begin = (iq * light_samples_per_epoch) / lines_per_epoch;
        //    int light_i_end = ((iq + 1) * light_samples_per_epoch) / lines_per_epoch;
        //    for (int i = light_i_begin; i < light_i_end; ++i) {
        //        size_t light_triangle_index;
        //        light_triangles_distribution.sample(random, light_triangle_index);
        //        LightTriangle const& lt = light_triangles[light_triangle_index];
        //        World::Triangle const& wtri = context.world.zone_trees[lt.zone_id].triangles[lt.triangle_index];
        //        Surface surface = test_material_surface(context.world, lt.zone_id, lt.triangle_index);
        //        Vec3 a = context.world.vertices[wtri.vertex_indexes[0]];
        //        Vec3 b = context.world.vertices[wtri.vertex_indexes[1]];
        //        Vec3 c = context.world.vertices[wtri.vertex_indexes[2]];
        //        float qa, qb, qc;
        //        random.uniform(qa);
        //        random.uniform(qb);
        //        random.uniform(qc);
        //        float u = fabsf(qa - qb);
        //        float v = (1.0f - u) * qc;
        //        Vec3 light_origin = a + u * (b - a) + v * (c - a);
        //        uint64_t path_node_index;
        //        PathNode& path_node = current_epoch_view.path_node_allocator.emplace(path_node_index);
        //        path_node.prev_node_index = path_origin_bit | (lt.zone_id << 32) | lt.triangle_index;
        //        path_node.node_level = 0;
        //        path_node.incoming_value_density = (1.0f / lt.probability_density) * Vec3{1.0f, 1.0f, 1.0f};
        //        path_node.incoming_direction = surface.normal;
        //        path_node.position = light_origin;
        //        path_node.surface = surface;
        //        {
        //            Vec3 light_dir;
        //            Vec3 emit_value;
        //            surface.sample_emission(random, light_dir, emit_value);
        //            path_node.scattered_value_density = elementwise_product(path_node.incoming_value_density, emit_value);
        //            inserter.schedule(
        //                lt.zone_id,
        //                light_origin,
        //                light_dir,
        //                make_ray_id(RayType::LightPath, current_epoch_view.current_epoch, path_node_index));
        //            current_epoch_view->use_count.fetch_add(1, std::memory_order_relaxed);
        //        }
        //        Vec3 direction_to_camera = context.camera.origin - light_origin;
        //        if (dot(direction_to_camera, surface.normal) > 0) {
        //            int ix, iy;
        //            if (context.camera.world_to_pixel(random, direction_to_camera, ix, iy)) {
        //                Vec3 dn = norm(direction_to_camera);
        //                Vec3 light_flux_density = path_node.surface.emission_flux_density(dn);
        //                Vec3 lens_normal = context.camera.forward;
        //                float lens_flux_density = context.camera.importance_flux_density(-dn);
        //                float geom = geometric(lens_normal, path_node.surface.normal, direction_to_camera);
        //                assign_elementwise_product(light_flux_density, path_node.incoming_value_density);
        //                Vec3 value = geom * lens_flux_density * light_flux_density;
        //                uint64_t connector_index;
        //                Connector& conn = current_epoch_view.connector_allocator.emplace(connector_index);
        //                conn.lens_path_node = World::invalid_index;
        //                conn.light_path_node = path_node_index;
        //                conn.ix = ix;
        //                conn.iy = iy;
        //                conn.value = value;
        //                inserter.schedule(
        //                    lt.zone_id,
        //                    light_origin,
        //                    direction_to_camera,
        //                    make_ray_id(RayType::Connector, current_epoch_view.current_epoch, connector_index),
        //                    0.0f, 1.0f);
        //                current_epoch_view->use_count.fetch_add(1, std::memory_order_relaxed);
        //            }
        //        }
        //        context.increment_weight_global(1.0f / (float)(width * height));
        //    }
        //    //int iy = (current_epoch_view->epoch_number % epochs_per_frame) * lines_per_epoch + iq;
        //    //if (iy < height) {
        //    //    for (int ix = 0; ix < width; ++ix) {
        //    //        uint64_t path_node_index;
        //    //        PathNode& path_node = current_epoch_view.path_node_allocator.emplace(path_node_index);
        //    //        path_node.prev_node_index = path_origin_bit | (iy * width + ix);
        //    //        path_node.node_level = 0;
        //    //        path_node.incoming_value_density = Vec3{1.0f, 1.0f, 1.0f};
        //    //        path_node.incoming_direction = context.camera.forward;
        //    //        path_node.position = context.camera.origin;
        //    //        path_node.surface = Surface::black(context.camera.forward);
        //    //        Vec3 lens_dir;
        //    //        context.camera.pixel_to_world(random, ix, iy, lens_dir);
        //    //        path_node.scattered_value_density = path_node.incoming_value_density;
        //    //        inserter.schedule(
        //    //            context.camera.zone_id,
        //    //            context.camera.origin,
        //    //            lens_dir,
        //    //            make_ray_id(RayType::LensPath, current_epoch_view.current_epoch, path_node_index));
        //    //        current_epoch_view->use_count.fetch_add(1, std::memory_order_relaxed);
        //    //        context.increment_accum_weight(ix, iy, 1.0f);
        //    //    }
        //    //}
        //}
        return work_performed;
    }
};

struct LoaderProcessor: Processor {
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

    LoaderProcessor(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
        : Processor(std::move(control_ptr))
        , context(context)
    {
        packet_count_threshold = 100000 / context.pipeline.get_params().ray_buffer_size;
        World::Tree const& tree = context.world.zone_trees[context.camera.zone_id];
        std::vector<float> lt_weights;
        float total_weight = 0.0f;
        for (size_t i = 0; i < tree.triangles.size(); ++i) {
            World::Triangle const& tri = tree.triangles[i];
            Surface surface = test_material_surface(context.world, context.camera.zone_id, i);
            float density = surface.emission.x + surface.emission.y + surface.emission.z;
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
        CurrentEpochView current_epoch_view(context);
        while (true) {
            if (context.pipeline.get_total_packet_count() >= packet_count_threshold) {
                break;
            }
            if (light_triangles.size() == 0) {
                break;
            }
            int iq;
            if (!current_epoch_view.advance_loader_progress(work_performed, iq, lines_per_epoch)) {
                break;
            }
            work_performed = true;
            int light_i_begin = (iq * light_samples_per_epoch) / lines_per_epoch;
            int light_i_end = ((iq + 1) * light_samples_per_epoch) / lines_per_epoch;
            for (int i = light_i_begin; i < light_i_end; ++i) {
                size_t light_triangle_index;
                light_triangles_distribution.sample(random, light_triangle_index);
                LightTriangle const& lt = light_triangles[light_triangle_index];
                World::Triangle const& wtri = context.world.zone_trees[lt.zone_id].triangles[lt.triangle_index];
                Surface surface = test_material_surface(context.world, lt.zone_id, lt.triangle_index);
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
                PathNode& path_node = current_epoch_view.path_node_allocator.emplace(path_node_index);
                path_node.prev_node_index = path_origin_bit | (lt.zone_id << 32) | lt.triangle_index;
                path_node.node_level = 0;
                path_node.incoming_value_density = (1.0f / lt.probability_density) * Vec3{1.0f, 1.0f, 1.0f};
                path_node.incoming_direction = surface.normal;
                path_node.position = light_origin;
                path_node.surface = surface;
                {
                    Vec3 light_dir;
                    Vec3 emit_value;
                    surface.sample_emission(random, light_dir, emit_value);
                    path_node.scattered_value_density = elementwise_product(path_node.incoming_value_density, emit_value);
                    inserter.schedule(
                        lt.zone_id,
                        light_origin,
                        light_dir,
                        make_ray_id(RayType::LightPath, current_epoch_view.current_epoch, path_node_index));
                    current_epoch_view->use_count.fetch_add(1, std::memory_order_relaxed);
                }
                Vec3 direction_to_camera = context.camera.origin - light_origin;
                if (dot(direction_to_camera, surface.normal) > 0) {
                    int ix, iy;
                    if (context.camera.world_to_pixel(random, direction_to_camera, ix, iy)) {
                        Vec3 dn = norm(direction_to_camera);
                        Vec3 light_flux_density = path_node.surface.emission_flux_density(dn);
                        Vec3 lens_normal = context.camera.forward;
                        float lens_flux_density = context.camera.importance_flux_density(-dn);
                        float geom = geometric(lens_normal, path_node.surface.normal, direction_to_camera);
                        assign_elementwise_product(light_flux_density, path_node.incoming_value_density);
                        Vec3 value = geom * lens_flux_density * light_flux_density;
                        uint64_t connector_index;
                        Connector& conn = current_epoch_view.connector_allocator.emplace(connector_index);
                        conn.lens_path_node = World::invalid_index;
                        conn.light_path_node = path_node_index;
                        conn.ix = ix;
                        conn.iy = iy;
                        conn.value = value;
                        inserter.schedule(
                            lt.zone_id,
                            light_origin,
                            direction_to_camera,
                            make_ray_id(RayType::Connector, current_epoch_view.current_epoch, connector_index),
                            0.0f, 1.0f);
                        current_epoch_view->use_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                context.increment_weight_global(1.0f / (float)(width * height));
            }
            //int iy = (current_epoch_view->epoch_number % epochs_per_frame) * lines_per_epoch + iq;
            //if (iy < height) {
            //    for (int ix = 0; ix < width; ++ix) {
            //        uint64_t path_node_index;
            //        PathNode& path_node = current_epoch_view.path_node_allocator.emplace(path_node_index);
            //        path_node.prev_node_index = path_origin_bit | (iy * width + ix);
            //        path_node.node_level = 0;
            //        path_node.incoming_value_density = Vec3{1.0f, 1.0f, 1.0f};
            //        path_node.incoming_direction = context.camera.forward;
            //        path_node.position = context.camera.origin;
            //        path_node.surface = Surface::black(context.camera.forward);
            //        Vec3 lens_dir;
            //        context.camera.pixel_to_world(random, ix, iy, lens_dir);
            //        path_node.scattered_value_density = path_node.incoming_value_density;
            //        inserter.schedule(
            //            context.camera.zone_id,
            //            context.camera.origin,
            //            lens_dir,
            //            make_ray_id(RayType::LensPath, current_epoch_view.current_epoch, path_node_index));
            //        current_epoch_view->use_count.fetch_add(1, std::memory_order_relaxed);
            //        context.increment_accum_weight(ix, iy, 1.0f);
            //    }
            //}
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
        //MainCamera camera = MainCamera::targeted(world, Vec3{108, -1150, 55}, Vec3{64, -1150, 236}, 0.1f);

        Context context{pauserequested, steprequested, world, pipeline, camera};
        context.ray_counter.store(0, std::memory_order_relaxed);
        context.accum_buffer = std::make_unique<std::atomic<uint64_t>[]>(width * height * 4);
        for (size_t i = 0; i < width * height * 4; ++i) {
            context.accum_buffer[i].store(0, std::memory_order_relaxed);
        }
        context.global_denominator_additive.store(0, std::memory_order_relaxed);

        std::shared_ptr<ProcessorControl> this_control_ptr = std::make_shared<ProcessorControl>();
        Scheduler::register_processor(std::make_shared<LoaderProcessor>(this_control_ptr, context));
        //Scheduler::register_processor(std::make_shared<ConnectorProcessor>(this_control_ptr, context));
        Scheduler::register_processor(std::make_shared<HandlerProcessor>(this_control_ptr, context));

        uint64_t start_time = GetTickCount64() + 15000;

        while (!rterminate.load(std::memory_order_relaxed)) {
            using namespace std::chrono_literals;

            std::this_thread::sleep_for(100ms);

            {
                auto& bits = bitbuf.Back();
                bits.resize(width * height * 4);
                uint8_t* pixels = bits.data();
                uint64_t global_den = context.global_denominator_additive.load(std::memory_order_relaxed);
                for (int iy = 0; iy < height; ++iy) {
                    uint8_t* line = pixels + iy * width * 4;
                    for (int ix = 0; ix < width; ++ix) {
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
                    "rt | %llu frames | packets: %llu | flushing",
                    context.total_frames_completed.load(std::memory_order_relaxed),
                    pipeline.get_total_packet_count());
            } else {
                uint64_t counter = context.ray_counter.load(std::memory_order_relaxed);
                uint64_t time = GetTickCount64();
                if (time <= start_time) {
                    context.ray_counter.store(0, std::memory_order_relaxed);
                    snprintf(
                        buf, sizeof(buf),
                        "rt | %llu frames | packets: %llu | warming up (%llus)",
                        context.total_frames_completed.load(std::memory_order_relaxed),
                        pipeline.get_total_packet_count(),
                        (start_time - time) / 1000);
                } else {
                    snprintf(
                        buf, sizeof(buf),
                        "rt | %llu frames | packets: %llu | Krays per second: %llu",
                        context.total_frames_completed.load(std::memory_order_relaxed),
                        pipeline.get_total_packet_count(),
                        counter / (time - start_time + 1));
                }
            }
            SendMessageTimeoutA(
                hwnd, WM_SETTEXT, (WPARAM)nullptr, (LPARAM)buf,
                SMTO_NORMAL, 100, nullptr);
        }

        this_control_ptr->set_dead();
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
    pauserequested.store(true, std::memory_order_relaxed);
    steprequested.store(true, std::memory_order_relaxed);
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

void RendererThread::Step()
{
    steprequested.store(true, std::memory_order_relaxed);
}
