#include "RendererThread.h"
#include "Color.h"
#include "RayPipeline.h"
#include "Random.h"
#include "Threading.h"
#include "World.h"
//#include <png.h>

//#pragma optimize( "", off )

constexpr int width = 1600;
constexpr int height = 900;
constexpr int pixels_per_epoch = 0x400;
constexpr int epochs_per_frame = (width * height + pixels_per_epoch - 1) / pixels_per_epoch;
constexpr int light_samples_per_epoch = pixels_per_epoch;
constexpr int connections_per_lens_sample = 4;
//constexpr int light_samples_per_epoch = 1;
//constexpr float exposure = 0.5f;
constexpr float exposure = 70.0f;
constexpr float accum_num_scale = 10000000.0f;
constexpr float accum_den_scale = 100.0f * width * height * epochs_per_frame;

constexpr float pi = 3.141593f;

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

static void tonemap(Color v, uint8_t* pixel)
{
    pixel[0] = tonemap_curve(v[2]);
    pixel[1] = tonemap_curve(v[1]);
    pixel[2] = tonemap_curve(v[0]);
    pixel[3] = 255;
}

struct Surface {
    Vec3 normal;
    Color albedo = Color{0, 0, 0};
    Color emission = Color{0, 0, 0};

    static Surface black(Vec3 normal) {
        Surface m;
        m.normal = normal;
        return m;
    }

    static Surface emissive(Vec3 normal, float ex, float ey, float ez) {
        Surface m;
        m.normal = normal;
        m.emission = Color{ex, ey, ez};
        return m;
    }

    static Surface diffuse(Vec3 normal, float ax, float ay, float az) {
        Surface m;
        m.normal = normal;
        m.albedo = Color{ax, ay, az};
        return m;
    }

    Color emission_flux_density(Vec3 lens_dir) const {
        return emission;
    }

    float emission_probability_density(Vec3 lens_dir) const {
        return 1.0f / pi;
    }

    void sample_emission(Random& random, Vec3& light_dir, Color& result_value) const {
        random.lambert(normal, light_dir);
        result_value = pi * emission;
    }

    float scatter_event_probability_from_lens(int path_length, Vec3 lens_dir) const {
        if (path_length <= 4) {
            return 1.0f;
        } else {
            float p = elementwise_foldl1(fmaxf, albedo);
            float q = 1.0f - p;
            return 1.0f - q*q;
        }
    }

    float scatter_event_probability_from_light(int path_length, Vec3 lens_dir) const {
        return scatter_event_probability_from_lens(path_length, lens_dir);
    }

    Color scatter_flux_density(Vec3 lens_dir, Vec3 light_dir) const {
        return (1.0f / pi) * albedo;
    }

    float scatter_probability_density(Vec3 lens_dir, Vec3 light_dir) const {
        return 1.0f / pi;
    }

    void sample_scatter_from_lens(Random& random, Vec3 lens_dir, Vec3& light_dir, Color& result_value) const {
        random.lambert(normal, light_dir);
        result_value = albedo;
    }

    void sample_scatter_from_light(Random& random, Vec3 light_dir, Vec3& lens_dir, Color& result_value) const {
        sample_scatter_from_lens(random, light_dir, lens_dir, result_value);
    }
};

Surface test_material_surface(World const& world, size_t triangle_index) {
    if (triangle_index != World::invalid_index) {
        World::Triangle const& tri = world.triangles[triangle_index];
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
                } else if (mat_name == "Engine.Texture'Botpack.Ammocount.AmmoCountBar'") {
                    return Surface::emissive(normal, 1, 1, 1);
                } else {
                    //return Surface::diffuse(normal, 0.02f, 0.02f, 0.02f);
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
        camera.zone_id = world.zone_id_at(origin);
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

    float ray_density(Vec3 direction) const {
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

struct LightTriangleDistribution {
    DiscreteDistribution distribution;
    std::vector<float> triangle_probability_density;

    LightTriangleDistribution() = default;
    LightTriangleDistribution(LightTriangleDistribution&&) = default;
    LightTriangleDistribution& operator=(LightTriangleDistribution&&) = default;

    LightTriangleDistribution(World const& world, std::vector<Surface> const& world_surfaces, MainCamera const& camera) {
        std::vector<float> lt_weights(world.triangles.size(), 0.0f);
        triangle_probability_density.resize(world.triangles.size(), 0.0f);
        bool any_triangle_chosen = false;
        for (size_t triangle_index = 0; triangle_index < world.triangles.size(); ++triangle_index) {
            World::Triangle const& tri = world.triangles[triangle_index];
            if (tri.zone_id == camera.zone_id) {
                Surface const& surface = world_surfaces[triangle_index];
                float density = elementwise_foldl1([](float a, float b) { return a + b; }, surface.emission);
                if (density > 0.0f && tri.area > 0.0f) {
                    lt_weights[triangle_index] = density * tri.area;
                    any_triangle_chosen = true;
                }
            }
        }
        if (!any_triangle_chosen) {
            lt_weights[0] = 1.0f;
        }
        distribution = DiscreteDistribution::make(lt_weights);
        for (size_t triangle_index = 0; triangle_index < world.triangles.size(); ++triangle_index) {
            World::Triangle const& tri = world.triangles[triangle_index];
            if (tri.area > 0.0f) {
                triangle_probability_density[triangle_index] = distribution.probability_of(triangle_index) / tri.area;
            }
        }
    }
};

struct PathNode {
    uint64_t prev_node_index;
    int path_length;
    int path_index;
    Color incoming_value_density;
    Vec3 incoming_direction;
    Vec3 position;
    size_t triangle_index;
    Color scattered_value_density;
};

struct Connector {
    uint64_t lens_path_node;
    uint64_t light_path_node;
    int ix;
    int iy;
    Color value;
    size_t expected_hit_triangle_index;
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
    std::shared_mutex mutex;
    Arena<PathNode> path_node_arena;
    Arena<Connector> connector_arena;
    AlignedBuffer light_sample_buffer;
    std::atomic<uint64_t> use_count;
    std::atomic<int> phase_one_progress;
    std::atomic<int> completed_light_paths_count;
    std::atomic<int> phase_two_progress;
    std::atomic<size_t> total_resets;
    std::atomic<size_t> total_phase_one_rays_sent;
    std::atomic<size_t> total_phase_two_rays_sent;
    size_t epoch_number;
    LightTriangleDistribution light_triangles_distribution;

    Epoch() {
        light_sample_buffer = AlignedBuffer(light_samples_per_epoch * sizeof(uint64_t));
        use_count.store(0, std::memory_order_relaxed);
        phase_one_progress.store(light_samples_per_epoch, std::memory_order_relaxed);
        completed_light_paths_count.store(light_samples_per_epoch, std::memory_order_relaxed);
        phase_two_progress.store(pixels_per_epoch, std::memory_order_relaxed);
        total_resets.store(0, std::memory_order_relaxed);
        total_phase_one_rays_sent.store(0, std::memory_order_relaxed);
        total_phase_two_rays_sent.store(0, std::memory_order_relaxed);
    }

    uint64_t& light_sample_at(int i) {
        return ((uint64_t*)light_sample_buffer.data())[i];
    }

    bool advance_phase_one_progress(int delta, int& iq, int limit) {
        iq = phase_one_progress.fetch_add(delta, std::memory_order_relaxed);
        if (iq < limit) {
            return true;
        } else {
            phase_one_progress.store(limit, std::memory_order_relaxed);
            return false;
        }
    }

    bool advance_phase_two_progress(int delta, int& iq, int limit) {
        iq = phase_two_progress.fetch_add(delta, std::memory_order_relaxed);
        if (iq < limit) {
            return true;
        } else {
            phase_two_progress.store(limit, std::memory_order_relaxed);
            return false;
        }
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

    AtomicCounterGuard(std::atomic<T>& counter, std::adopt_lock_t, T delta = 1) {
        _counter_ptr = &counter;
        _delta = delta;
    }

    AtomicCounterGuard(AtomicCounterGuard&& other) {
        _counter_ptr = other._counter_ptr;
        _delta = other._delta;
        other._delta = 0;
    }

    AtomicCounterGuard& operator=(AtomicCounterGuard&& other) {
        unlock();
        _counter_ptr = other._counter_ptr;
        _delta = other._delta;
        other._delta = 0;
        return *this;
    }

    ~AtomicCounterGuard() {
        unlock();
    }

    void unlock(std::memory_order decrement_order = std::memory_order_release) {
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
    static constexpr uint8_t epoch_count = 24;

    std::atomic<bool>& pauserequested;
    std::atomic<bool>& steprequested;
    World const& world;
    RayPipeline& pipeline;
    MainCamera const& camera;
    std::vector<Surface> world_surfaces;
    Epoch epochs[epoch_count];
    std::mutex next_epoch_number_mutex;
    size_t next_epoch_number;
    std::atomic<size_t> total_frames_completed;
    std::atomic<uint64_t> current_ray_count;
    std::atomic<uint64_t> ray_counter;
    std::unique_ptr<std::atomic<uint64_t>[]> overlay_accum_buffer;
    std::unique_ptr<std::atomic<uint64_t>[]> additive_accum_buffer;
    std::atomic<uint64_t> additive_denominator;
    size_t packet_count_threshold;

    Context(std::atomic<bool>& pauserequested, std::atomic<bool>& steprequested, World const& world, RayPipeline& pipeline, MainCamera const& camera)
        : pauserequested(pauserequested)
        , steprequested(steprequested)
        , world(world)
        , pipeline(pipeline)
        , camera(camera)
    {
        world_surfaces.resize(world.triangles.size());
        for (size_t i = 0; i < world_surfaces.size(); ++i) {
            world_surfaces[i] = test_material_surface(world, i);
        }
        total_frames_completed.store(0, std::memory_order_relaxed);
        current_ray_count.store(0, std::memory_order_relaxed);
        ray_counter.store(0, std::memory_order_relaxed);
        next_epoch_number = 0;
        overlay_accum_buffer = std::make_unique<std::atomic<uint64_t>[]>(width * height * 4);
        for (size_t i = 0; i < width * height * 4; ++i) {
            overlay_accum_buffer[i].store(0, std::memory_order_relaxed);
        }
        additive_accum_buffer = std::make_unique<std::atomic<uint64_t>[]>(width * height * 3);
        for (size_t i = 0; i < width * height * 3; ++i) {
            additive_accum_buffer[i].store(0, std::memory_order_relaxed);
        }
        additive_denominator.store(0, std::memory_order_relaxed);
        packet_count_threshold = 5000;
    }

    bool is_paused() {
        return pauserequested.load(std::memory_order_relaxed) && !steprequested.load(std::memory_order_relaxed);
    }

    void consume_step_request() {
        steprequested.store(false, std::memory_order_relaxed);
    }

    Surface triangle_surface(size_t triangle_index) {
        if (triangle_index != World::invalid_index) {
            return world_surfaces[triangle_index];
        } else {
            //emit = Vec3{100.00f, 0.00f, 100.00f};
            return Surface::black(Vec3{0.0f, 0.0f, 1.0f});
        }
    }

    Color color_at(int ix, int iy) {
        int pixel_i = iy * width + ix;
        std::atomic<uint64_t>* overlay_accum_ptr = overlay_accum_buffer.get() + 4 * pixel_i;
        float ow = overlay_accum_ptr[3].load(std::memory_order_relaxed);
        Color result{0, 0, 0};
        if (ow != 0) {
            float scale = (accum_den_scale / accum_num_scale) / ow;
            result[0] = scale * (float)overlay_accum_ptr[0].load(std::memory_order_relaxed);
            result[1] = scale * (float)overlay_accum_ptr[1].load(std::memory_order_relaxed);
            result[2] = scale * (float)overlay_accum_ptr[2].load(std::memory_order_relaxed);
        }
        std::atomic<uint64_t>* additive_accum_ptr = additive_accum_buffer.get() + 3 * pixel_i;
        float aw = additive_denominator.load(std::memory_order_relaxed);
        if (aw != 0) {
            float scale = (accum_den_scale / accum_num_scale) / aw;
            result[0] += scale * (float)additive_accum_ptr[0].load(std::memory_order_relaxed);
            result[1] += scale * (float)additive_accum_ptr[1].load(std::memory_order_relaxed);
            result[2] += scale * (float)additive_accum_ptr[2].load(std::memory_order_relaxed);
        }
        return result;
    }

    void increment_overlay_accum_color(int ix, int iy, Color value) {
        uint64_t vr = value[0] * accum_num_scale + 0.5f;
        uint64_t vg = value[1] * accum_num_scale + 0.5f;
        uint64_t vb = value[2] * accum_num_scale + 0.5f;
        std::atomic<uint64_t>* overlay_accum_ptr = overlay_accum_buffer.get() + 4 * width * iy + 4 * ix;
        if (vr != 0) overlay_accum_ptr[0].fetch_add(vr, std::memory_order_relaxed);
        if (vg != 0) overlay_accum_ptr[1].fetch_add(vg, std::memory_order_relaxed);
        if (vb != 0) overlay_accum_ptr[2].fetch_add(vb, std::memory_order_relaxed);
    }

    void increment_overlay_accum_weight(int ix, int iy, float weight) {
        uint64_t vw = weight * accum_den_scale + 0.5f;
        std::atomic<uint64_t>* overlay_accum_ptr = overlay_accum_buffer.get() + 4 * width * iy + 4 * ix;
        if (vw != 0) overlay_accum_ptr[3].fetch_add(vw, std::memory_order_relaxed);
    }

    void increment_additive_accum_color(int ix, int iy, Color value) {
        uint64_t vr = value[0] * accum_num_scale + 0.5f;
        uint64_t vg = value[1] * accum_num_scale + 0.5f;
        uint64_t vb = value[2] * accum_num_scale + 0.5f;
        std::atomic<uint64_t>* additive_accum_ptr = additive_accum_buffer.get() + 3 * width * iy + 3 * ix;
        if (vr != 0) additive_accum_ptr[0].fetch_add(vr, std::memory_order_relaxed);
        if (vg != 0) additive_accum_ptr[1].fetch_add(vg, std::memory_order_relaxed);
        if (vb != 0) additive_accum_ptr[2].fetch_add(vb, std::memory_order_relaxed);
    }

    void increment_additive_weight(float weight) {
        uint64_t vw = weight * accum_den_scale + 0.5f;
        if (vw != 0) additive_denominator.fetch_add(vw, std::memory_order_relaxed);
    }
};

struct EpochView {
    Epoch* epoch_ptr;
    std::shared_lock<std::shared_mutex> epoch_mutex_guard;
    AtomicCounterGuard<uint64_t> epoch_counter_guard;
    Arena<PathNode>::View path_node_arena_view;
    Arena<Connector>::View connector_arena_view;
    Arena<PathNode>::Allocator path_node_allocator;
    Arena<Connector>::Allocator connector_allocator;

    EpochView() {
        epoch_ptr = nullptr;
    }

    EpochView(Epoch& epoch_object)
        : epoch_ptr(&epoch_object)
    {
        initialize();
    }

    EpochView(Epoch& epoch_object, std::defer_lock_t)
        : epoch_ptr(&epoch_object)
    {
    }

    EpochView(EpochView&& other) = default;
    EpochView& operator=(EpochView&& other) = default;

    ~EpochView() {
        unlock();
    }

    void initialize() {
        if (!epoch_counter_guard) {
            epoch_mutex_guard = std::shared_lock<std::shared_mutex>(epoch_ptr->mutex);
            epoch_counter_guard = AtomicCounterGuard<uint64_t>(epoch_ptr->use_count);
            path_node_arena_view = epoch_ptr->path_node_arena.view();
            connector_arena_view = epoch_ptr->connector_arena.view();
            path_node_allocator = epoch_ptr->path_node_arena.allocator();
            connector_allocator = epoch_ptr->connector_arena.allocator();
        }
    }

    void unlock() {
        if (epoch_counter_guard) {
            path_node_allocator.release();
            connector_allocator.release();
            epoch_counter_guard.unlock();
            epoch_mutex_guard.unlock();
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
        return path_node_allocator.emplace(index, std::forward<Ts>(args)...);
    }

    template<typename... Ts>
    Connector& emplace_connector(uint64_t& index, Ts&&... args) {
        return connector_allocator.emplace(index, std::forward<Ts>(args)...);
    }

    Epoch* operator->() {
        return epoch_ptr;
    }
};

namespace {
    float geometric(Vec3 na, Vec3 nb, Vec3 d) {
        float dsqr = dotsqr(d);
        return -dot(na, d) * dot(nb, d) / (dsqr * dsqr);
    }

    PathNode camera_path_node(MainCamera const& camera, int ix, int iy) {
        PathNode path_node;
        path_node.prev_node_index = path_origin_bit | (iy * width + ix);
        path_node.path_length = 1;
        path_node.incoming_value_density = Color{1.0f, 1.0f, 1.0f};
        path_node.incoming_direction = camera.forward;
        path_node.position = camera.origin;
        path_node.triangle_index = World::invalid_index;
        return path_node;
    }

    void get_lens_path_origin_pixel(EpochView& epoch_view, PathNode const& base_path_node, int& ix, int& iy) {
        PathNode const* iter_node_ptr = &base_path_node;
        while ((iter_node_ptr->prev_node_index & path_origin_bit) == 0) {
            iter_node_ptr = &epoch_view.path_node_at(iter_node_ptr->prev_node_index);
        }
        uint64_t pixel = iter_node_ptr->prev_node_index & ~path_origin_bit;
        ix = pixel % width;
        iy = pixel / width;
    }

    bool is_path_enabled(int lens_path_length, int light_path_length) {
        return true;
        //return lens_path_length + light_path_length == 3;
    }

    bool is_path_enabled_and_recorded(int lens_path_length, int light_path_length) {
        return is_path_enabled(lens_path_length, light_path_length) &&
            (true || lens_path_length == 3 && light_path_length == 0);
    }

    constexpr int max_lens_path_length = 1000;
    constexpr int max_light_path_length = 1000;

    float lens_path_alt_weight(Context& context, EpochView& epoch_view, float probability_base, PathNode const& initial_lens_node, int initial_light_path_length, Vec3 initial_direction) {
        float total_weight = 0.0f;
        PathNode const* current_node_ptr = &initial_lens_node;
        Surface current_node_surface = context.triangle_surface(initial_lens_node.triangle_index);
        int current_light_path_length = initial_light_path_length + 1;
        Vec3 current_incoming_dir = -initial_direction;
        float current_probability = probability_base;
        while (!(current_node_ptr->prev_node_index & path_origin_bit)) {
            PathNode const& prev_node = epoch_view.path_node_at(current_node_ptr->prev_node_index);
            Surface prev_node_surface = context.triangle_surface(prev_node.triangle_index);
            float current_to_prev_geom;
            float prev_node_full_scatter_prob;
            if (prev_node.prev_node_index & path_origin_bit) {
                current_to_prev_geom = geometric(context.camera.forward, current_node_surface.normal, current_node_ptr->position - prev_node.position);
                prev_node_full_scatter_prob =
                    ((float)pixels_per_epoch / light_samples_per_epoch) *
                    connections_per_lens_sample *
                    context.camera.ray_density(-current_node_ptr->incoming_direction);
            } else {
                current_to_prev_geom = geometric(prev_node_surface.normal, current_node_surface.normal, current_node_ptr->position - prev_node.position);
                prev_node_full_scatter_prob =
                    prev_node_surface.scatter_event_probability_from_lens(prev_node.path_length, prev_node.incoming_direction) *
                    prev_node_surface.scatter_probability_density(prev_node.incoming_direction, -current_node_ptr->incoming_direction);
            }
            if (is_path_enabled(prev_node.path_length, current_light_path_length)) {
                if (current_to_prev_geom > 0.0f) {
                    float to_prev_prob = current_probability / (current_to_prev_geom * prev_node_full_scatter_prob);
                    total_weight += to_prev_prob * to_prev_prob;
                }
            }
            float current_node_full_scatter_prob;
            if (current_light_path_length == 1) {
                current_node_full_scatter_prob =
                    current_node_surface.emission_probability_density(current_incoming_dir);
            } else {
                current_node_full_scatter_prob =
                    current_node_surface.scatter_event_probability_from_light(current_light_path_length, current_node_ptr->incoming_direction) *
                    current_node_surface.scatter_probability_density(current_incoming_dir, current_node_ptr->incoming_direction);
            }
            current_probability *= current_node_full_scatter_prob / prev_node_full_scatter_prob;
            current_node_surface = prev_node_surface;
            current_light_path_length += 1;
            current_incoming_dir = -current_node_ptr->incoming_direction;
            current_node_ptr = &prev_node;
        }
        return total_weight;
    }

    float light_path_alt_weight(Context& context, EpochView& epoch_view, float probability_base, int initial_lens_path_length, PathNode const& initial_light_node, Vec3 initial_direction) {
        float total_weight = 0.0f;
        PathNode const* current_node_ptr = &initial_light_node;
        Surface current_node_surface = context.triangle_surface(initial_light_node.triangle_index);
        int current_lens_path_length = initial_lens_path_length + 1;
        Vec3 current_incoming_dir = initial_direction;
        float current_probability = probability_base;
        while (!(current_node_ptr->prev_node_index & path_origin_bit)) {
            PathNode const& prev_node = epoch_view.path_node_at(current_node_ptr->prev_node_index);
            Surface prev_node_surface = context.triangle_surface( prev_node.triangle_index);
            float current_to_prev_geom = geometric(prev_node_surface.normal, current_node_surface.normal, current_node_ptr->position - prev_node.position);
            float prev_node_full_scatter_prob;
            if (prev_node.prev_node_index & path_origin_bit) {
                prev_node_full_scatter_prob =
                    prev_node_surface.emission_probability_density(-current_node_ptr->incoming_direction);
            } else {
                prev_node_full_scatter_prob =
                    prev_node_surface.scatter_event_probability_from_lens(prev_node.path_length, prev_node.incoming_direction) *
                    prev_node_surface.scatter_probability_density(prev_node.incoming_direction, -current_node_ptr->incoming_direction);
            }
            if (is_path_enabled(current_lens_path_length, prev_node.path_length)) {
                if (current_to_prev_geom > 0.0f) {
                    float to_prev_prob = current_probability / (current_to_prev_geom * prev_node_full_scatter_prob);
                    total_weight += to_prev_prob * to_prev_prob;
                }
            }
            float current_node_full_scatter_prob =
                current_node_surface.scatter_event_probability_from_light(current_lens_path_length, current_node_ptr->incoming_direction) *
                current_node_surface.scatter_probability_density(current_node_ptr->incoming_direction, current_incoming_dir);
            current_probability *= current_node_full_scatter_prob / prev_node_full_scatter_prob;
            current_node_surface = prev_node_surface;
            current_lens_path_length += 1;
            current_incoming_dir = -current_node_ptr->incoming_direction;
            current_node_ptr = &prev_node;
        }
        if (is_path_enabled(current_lens_path_length, 0)) {
            size_t origin_light_triangle_index = current_node_ptr->prev_node_index & ~path_origin_bit;
            float origin_probability_density =
                connections_per_lens_sample *
                epoch_view->light_triangles_distribution.triangle_probability_density[origin_light_triangle_index];
            if (origin_probability_density > 0.0f) {
                float to_prev_prob = current_probability / origin_probability_density;
                total_weight += to_prev_prob * to_prev_prob;
            }
        }
        return total_weight;
    }

    bool get_connector_value(Context& context, EpochView& epoch_view, PathNode const& lens_node, PathNode const& light_node, Vec3& delta, Color& value) {
        if (is_path_enabled_and_recorded(lens_node.path_length, light_node.path_length)) {
            Surface lens_node_surface = context.triangle_surface(lens_node.triangle_index);
            Surface light_node_surface = context.triangle_surface(light_node.triangle_index);
            delta = lens_node.position - light_node.position;
            if (dot(delta, lens_node_surface.normal) < 0 && dot(delta, light_node_surface.normal) > 0) {
                Vec3 dn = norm(delta);
                Color light_flux_density;
                if (light_node.prev_node_index & path_origin_bit) {
                    light_flux_density = light_node_surface.emission_flux_density(dn);
                } else {
                    light_flux_density = light_node_surface.scatter_flux_density(dn, light_node.incoming_direction);
                }
                Color lens_flux_density = lens_node_surface.scatter_flux_density(lens_node.incoming_direction, -dn);
                float geom = geometric(lens_node_surface.normal, light_node_surface.normal, delta);
                light_flux_density *= light_node.incoming_value_density;
                lens_flux_density *= lens_node.incoming_value_density;
                float total_weight = 1.0f;
                float lens_alt_probability_base;
                if (light_node.prev_node_index & path_origin_bit) {
                    lens_alt_probability_base =
                        geom *
                        light_node_surface.emission_probability_density(dn);
                } else {
                    lens_alt_probability_base =
                        geom *
                        light_node_surface.scatter_event_probability_from_light(light_node.path_length, light_node.incoming_direction) *
                        light_node_surface.scatter_probability_density(dn, light_node.incoming_direction);
                }
                if (lens_alt_probability_base > 0.0f) {
                    total_weight += lens_path_alt_weight(context, epoch_view, lens_alt_probability_base, lens_node, light_node.path_length, dn);
                }
                float light_alt_probability_base =
                    geom *
                    lens_node_surface.scatter_event_probability_from_lens(lens_node.path_length, lens_node.incoming_direction) *
                    lens_node_surface.scatter_probability_density(lens_node.incoming_direction, -dn);
                if (light_alt_probability_base > 0.0f) {
                    total_weight += light_path_alt_weight(context, epoch_view, light_alt_probability_base, lens_node.path_length, light_node, dn);
                }
                value = (geom / total_weight) * (lens_flux_density * light_flux_density);
                return true;
            }
        }
        return false;
    }

    bool get_camera_connector_params(Random& random, Context& context, EpochView& epoch_view, PathNode const& light_node, Vec3& delta, int& ix, int& iy, Color& value) {
        if (is_path_enabled_and_recorded(1, light_node.path_length)) {
            Surface light_node_surface = context.triangle_surface(light_node.triangle_index);
            delta = context.camera.origin - light_node.position;
            if (dot(delta, context.camera.forward) < 0 && dot(delta, light_node_surface.normal) > 0) {
                if (context.camera.world_to_pixel(random, delta, ix, iy)) {
                    Vec3 dn = norm(delta);
                    Color light_flux_density;
                    if (light_node.prev_node_index & path_origin_bit) {
                        light_flux_density = light_node_surface.emission_flux_density(dn);
                    } else {
                        light_flux_density = light_node_surface.scatter_flux_density(dn, light_node.incoming_direction);
                    }
                    float camera_ray_density = context.camera.ray_density(-dn);
                    float geom = geometric(context.camera.forward, light_node_surface.normal, delta);
                    light_flux_density *= light_node.incoming_value_density;
                    float total_weight = 1.0f;
                    float light_alt_probability_base =
                        geom *
                        ((float)pixels_per_epoch / light_samples_per_epoch) *
                        connections_per_lens_sample *
                        camera_ray_density;
                    if (light_alt_probability_base > 0.0f) {
                        total_weight += light_path_alt_weight(context, epoch_view, light_alt_probability_base, 1, light_node, dn);
                    }
                    value = (geom * camera_ray_density / total_weight) * light_flux_density;
                    return true;
                }
            }
        }
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
        while (true) {
            std::unique_ptr<RayPipeline::RayPacket> packet_ptr = context.pipeline.pop_completed_packet();
            if (!packet_ptr) {
                break;
            }
            work_performed = true;
            EpochView epoch_views[Context::epoch_count];
            for (size_t epoch = 0; epoch < Context::epoch_count; ++epoch) {
                epoch_views[epoch] = EpochView(context.epochs[epoch], std::defer_lock);
            }
            for (int i = 0; i < packet_ptr->ray_count; ++i) {
                RayPipeline::RayData ray_data = packet_ptr->extract_ray_data(i);
                RayType ray_type;
                uint8_t epoch;
                uint64_t data_index;
                parse_ray_id(ray_data.extra_data, ray_type, epoch, data_index);
                epoch_views[epoch].initialize();
                switch (ray_type) {
                case RayType::LensPath:
                {
                    bool path_continued = false;
                    PathNode& prev_path_node = epoch_views[epoch].path_node_at(data_index);
                    if (ray_data.hit_world_triangle_index != World::invalid_index) {
                        int ix, iy;
                        get_lens_path_origin_pixel(epoch_views[epoch], prev_path_node, ix, iy);
                        uint64_t next_path_node_index;
                        PathNode& next_path_node = epoch_views[epoch].emplace_path_node(next_path_node_index);
                        next_path_node.prev_node_index = data_index;
                        next_path_node.path_length = prev_path_node.path_length + 1;
                        next_path_node.path_index = prev_path_node.path_index;
                        next_path_node.incoming_value_density = prev_path_node.scattered_value_density;
                        next_path_node.incoming_direction = -norm(ray_data.direction);
                        next_path_node.position = ray_data.origin + ray_data.max_param * ray_data.direction;
                        next_path_node.triangle_index = ray_data.hit_world_triangle_index;
                        Surface hit_surface = context.triangle_surface(next_path_node.triangle_index);
                        if (is_path_enabled_and_recorded(next_path_node.path_length, 0)) {
                            Color direct_emission = hit_surface.emission_flux_density(next_path_node.incoming_direction);
                            if (direct_emission[0] != 0.0f || direct_emission[1] != 0.0f || direct_emission[2] != 0.0f) {
                                float total_weight = 1.0f;
                                if (next_path_node.path_length > 2) {
                                    float base_probability =
                                        connections_per_lens_sample *
                                        epoch_views[epoch]->light_triangles_distribution.triangle_probability_density[next_path_node.triangle_index];
                                    if (base_probability != 0.0f) {
                                        total_weight += lens_path_alt_weight(context, epoch_views[epoch], base_probability, next_path_node, 0, next_path_node.incoming_direction);
                                    }
                                }
                                Color value = (1.0f / total_weight) * (direct_emission * next_path_node.incoming_value_density);
                                context.increment_overlay_accum_color(ix, iy, value);
                            }
                        }
                        for (int conn_i = 0; conn_i < connections_per_lens_sample; ++conn_i) {
                            uint32_t light_path_index;
                            random.uniform_below(light_samples_per_epoch, light_path_index);
                            size_t light_node_index = epoch_views[epoch]->light_sample_at(light_path_index);
                            while (!(light_node_index & path_origin_bit)) {
                                PathNode& light_node = epoch_views[epoch].path_node_at(light_node_index);
                                Vec3 delta;
                                Color value;
                                if (get_connector_value(context, epoch_views[epoch], next_path_node, light_node, delta, value)) {
                                    uint64_t connector_index;
                                    Connector& conn = epoch_views[epoch].emplace_connector(connector_index);
                                    conn.lens_path_node = next_path_node_index;
                                    conn.light_path_node = light_node_index;
                                    conn.ix = ix;
                                    conn.iy = iy;
                                    conn.value = (1.0f / (float)connections_per_lens_sample) * value;
                                    conn.expected_hit_triangle_index = next_path_node.triangle_index;
                                    if (epoch_views[epoch]->use_count.fetch_add(1, std::memory_order_acquire) == 0) {
                                        throw "bad refcount";
                                    }
                                    context.current_ray_count.fetch_add(1, std::memory_order_relaxed);
                                    inserter.schedule(
                                        packet_ptr->zone_id,
                                        light_node.position,
                                        delta,
                                        make_ray_id(RayType::Connector, epoch, connector_index),
                                        0.0f, 1.0f);
                                }
                                light_node_index = light_node.prev_node_index;
                            }
                        }
                        size_t target_zone_id = packet_ptr->zone_id;
                        Vec3 in_dir = next_path_node.incoming_direction;
                        float scatter_event_probability = hit_surface.scatter_event_probability_from_lens(next_path_node.path_length, in_dir);
                        bool scatter_pass;
                        random.p(scatter_event_probability, scatter_pass);
                        if (next_path_node.path_length >= max_lens_path_length) {
                            scatter_pass = false;
                        }
                        if (scatter_pass) {
                            Vec3 out_dir;
                            Color scatter_value;
                            hit_surface.sample_scatter_from_lens(random, in_dir, out_dir, scatter_value);
                            next_path_node.scattered_value_density = (1.0f / scatter_event_probability) * (next_path_node.incoming_value_density * scatter_value);
                            Vec3 new_origin = ray_data.origin + (ray_data.max_param + 0.0f) * ray_data.direction;
                            context.current_ray_count.fetch_add(1, std::memory_order_relaxed);
                            inserter.schedule(
                                target_zone_id,
                                new_origin,
                                out_dir,
                                make_ray_id(RayType::LensPath, epoch, next_path_node_index));
                            path_continued = true;
                        }
                    }
                    if (!path_continued) {
                        epoch_views[epoch]->use_count.fetch_sub(1, std::memory_order_release);
                    }
                    break;
                }
                case RayType::LightPath:
                {
                    bool path_continued = false;
                    PathNode& prev_path_node = epoch_views[epoch].path_node_at(data_index);
                    uint64_t next_path_node_index = World::invalid_index;
                    if (ray_data.hit_world_triangle_index != World::invalid_index) {
                        PathNode& next_path_node = epoch_views[epoch].emplace_path_node(next_path_node_index);
                        next_path_node.prev_node_index = data_index;
                        next_path_node.path_length = prev_path_node.path_length + 1;
                        next_path_node.path_index = prev_path_node.path_index;
                        next_path_node.incoming_value_density = prev_path_node.scattered_value_density;
                        next_path_node.incoming_direction = -norm(ray_data.direction);
                        next_path_node.position = ray_data.origin + ray_data.max_param * ray_data.direction;
                        next_path_node.triangle_index = ray_data.hit_world_triangle_index;
                        {
                            int ix, iy;
                            Vec3 direction_to_camera;
                            Color value;
                            if (get_camera_connector_params(random, context, epoch_views[epoch], next_path_node, direction_to_camera, ix, iy, value)) {
                                uint64_t connector_index;
                                Connector& conn = epoch_views[epoch].emplace_connector(connector_index);
                                conn.lens_path_node = World::invalid_index;
                                conn.light_path_node = next_path_node_index;
                                conn.ix = ix;
                                conn.iy = iy;
                                conn.value = value;
                                conn.expected_hit_triangle_index = World::invalid_index;
                                inserter.schedule(
                                    packet_ptr->zone_id,
                                    next_path_node.position,
                                    direction_to_camera,
                                    make_ray_id(RayType::Connector, epoch, connector_index),
                                    0.0f, 1.0f);
                                epoch_views[epoch]->use_count.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                        Surface hit_surface = context.triangle_surface(next_path_node.triangle_index);
                        size_t target_zone_id = packet_ptr->zone_id;
                        Vec3 in_dir = next_path_node.incoming_direction;
                        float scatter_event_probability = hit_surface.scatter_event_probability_from_light(next_path_node.path_length, in_dir);
                        scatter_event_probability *= 0.1f;
                        bool scatter_pass;
                        random.p(scatter_event_probability, scatter_pass);
                        if (next_path_node.path_length >= max_light_path_length) {
                            scatter_pass = false;
                        }
                        if (scatter_pass) {
                            Vec3 out_dir;
                            Color scatter_value;
                            hit_surface.sample_scatter_from_light(random, in_dir, out_dir, scatter_value);
                            next_path_node.scattered_value_density = (1.0f / scatter_event_probability) * (next_path_node.incoming_value_density * scatter_value);
                            Vec3 new_origin = ray_data.origin + (ray_data.max_param + 0.0f) * ray_data.direction;
                            context.current_ray_count.fetch_add(1, std::memory_order_relaxed);
                            inserter.schedule(
                                target_zone_id,
                                new_origin,
                                out_dir,
                                make_ray_id(RayType::LightPath, epoch, next_path_node_index));
                            path_continued = true;
                        }
                    }
                    if (!path_continued) {
                        epoch_views[epoch]->use_count.fetch_sub(1, std::memory_order_release);
                        uint64_t final_node_index;
                        if (next_path_node_index != World::invalid_index) {
                            final_node_index = next_path_node_index;
                        } else {
                            final_node_index = data_index;
                        }
                        epoch_views[epoch]->light_sample_at(prev_path_node.path_index) = final_node_index;
                        epoch_views[epoch]->completed_light_paths_count.fetch_add(1, std::memory_order_release);
                    }
                    break;
                }
                case RayType::Connector:
                {
                    Connector& conn = epoch_views[epoch].connector_at(data_index);
                    bool hit_nothing = ray_data.hit_world_triangle_index == World::invalid_index;
                    bool hit_expected = ray_data.hit_world_triangle_index == conn.expected_hit_triangle_index;
                    if (hit_nothing || hit_expected) {
                        size_t lens_path_length = 1;
                        if (conn.lens_path_node != World::invalid_index) {
                            lens_path_length = epoch_views[epoch].path_node_at(conn.lens_path_node).path_length;
                        }
                        if (lens_path_length == 1) {
                            context.increment_additive_accum_color(conn.ix, conn.iy, conn.value);
                        } else {
                            context.increment_overlay_accum_color(conn.ix, conn.iy, conn.value);
                        }
                    }
                    epoch_views[epoch]->use_count.fetch_sub(1, std::memory_order_release);
                    break;
                }
                default:
                    throw std::runtime_error("invalid ray type");
                }
            }
            context.ray_counter.fetch_add(packet_ptr->ray_count, std::memory_order_relaxed);
            context.current_ray_count.fetch_sub(packet_ptr->ray_count, std::memory_order_relaxed);
            context.pipeline.collect_ray_packet_spare(std::move(packet_ptr));
        }
        return work_performed;
    }
};

struct PhaseTwoProcessor: Processor {
    Context& context;
    Random random;
    int ray_packet_size;

    PhaseTwoProcessor(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
        : Processor(std::move(control_ptr))
        , context(context)
    {
        ray_packet_size = context.pipeline.get_params().ray_buffer_size;
    }

    virtual bool has_pending_work_impl() override {
        return context.pipeline.get_total_packet_count() < context.packet_count_threshold;
    }

    virtual bool work_impl() override {
        if (context.pipeline.get_total_packet_count() >= context.packet_count_threshold) {
            return false;
        }
        bool work_performed = false;
        RayPipeline::Inserter inserter = context.pipeline.inserter();
        bool any_epoch_processed = true;
        while (any_epoch_processed) {
            any_epoch_processed = false;
            for (size_t epoch = 0; epoch < Context::epoch_count; ++epoch) {
                EpochView epoch_view(context.epochs[epoch]);
                if (epoch_view->completed_light_paths_count.load(std::memory_order_acquire) >= light_samples_per_epoch) {
                    while (true) {
                        if (context.pipeline.get_total_packet_count() >= context.packet_count_threshold) {
                            break;
                        }
                        int iq;
                        if (!epoch_view->advance_phase_two_progress(ray_packet_size, iq, pixels_per_epoch)) {
                            break;
                        }
                        work_performed = true;
                        any_epoch_processed = true;
                        int pixel_i_begin = iq * epochs_per_frame + (epoch_view->epoch_number % epochs_per_frame);
                        int pixel_i_end = pixel_i_begin + ray_packet_size * epochs_per_frame;
                        if (pixel_i_end > width * height) {
                            pixel_i_end = width * height;
                        }
                        for (int pixel_i = pixel_i_begin; pixel_i < pixel_i_end; pixel_i += epochs_per_frame) {
                            int ix = pixel_i % width;
                            int iy = pixel_i / width;
                            uint64_t path_node_index;
                            PathNode& path_node = epoch_view.emplace_path_node(path_node_index);
                            path_node = camera_path_node(context.camera, ix, iy);
                            path_node.path_index = iy * width + ix;
                            Vec3 lens_dir;
                            context.camera.pixel_to_world(random, ix, iy, lens_dir);
                            path_node.scattered_value_density = path_node.incoming_value_density;
                            if (epoch_view->use_count.fetch_add(1, std::memory_order_acquire) == 0) {
                                throw "bad refcount";
                            }
                            context.current_ray_count.fetch_add(1, std::memory_order_relaxed);
                            epoch_view->total_phase_two_rays_sent.fetch_add(1, std::memory_order_relaxed);
                            inserter.schedule(
                                context.camera.zone_id,
                                context.camera.origin,
                                lens_dir,
                                make_ray_id(RayType::LensPath, epoch, path_node_index));
                            context.increment_overlay_accum_weight(ix, iy, 1.0f);
                        }
                        if (iq >= pixels_per_epoch - ray_packet_size) {
                            if ((epoch_view->epoch_number + 1) % epochs_per_frame == 0) {
                                context.total_frames_completed.fetch_add(1, std::memory_order_relaxed);
                            }
                            epoch_view->use_count.fetch_sub(1, std::memory_order_release);
                        }
                    }
                }
            }
        }
        return work_performed;
    }
};

struct PhaseOneProcessor: Processor {
    Context& context;
    Random random;
    int ray_packet_size;

    PhaseOneProcessor(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
        : Processor(std::move(control_ptr))
        , context(context)
    {
        ray_packet_size = context.pipeline.get_params().ray_buffer_size;
    }

    virtual bool has_pending_work_impl() override {
        return context.pipeline.get_total_packet_count() < context.packet_count_threshold;
    }

    virtual bool work_impl() override {
        if (context.pipeline.get_total_packet_count() >= context.packet_count_threshold) {
            return false;
        }
        bool work_performed = false;
        RayPipeline::Inserter inserter = context.pipeline.inserter();
        bool any_epoch_processed = true;
        while (any_epoch_processed) {
            any_epoch_processed = false;
            for (size_t epoch = 0; epoch < Context::epoch_count; ++epoch) {
                EpochView epoch_view(context.epochs[epoch], std::defer_lock);
                if (epoch_view->use_count.load(std::memory_order_relaxed) == 0) {
                    if (auto epoch_exclusive_guard = std::unique_lock<std::shared_mutex>(epoch_view->mutex, std::try_to_lock)) {
                        if (epoch_view->use_count.load(std::memory_order_relaxed) == 0) {
                            size_t new_epoch_number;
                            {
                                std::lock_guard<std::mutex> next_epoch_number_guard(context.next_epoch_number_mutex);
                                if (context.next_epoch_number % epochs_per_frame == 0) {
                                    if (context.pauserequested.load(std::memory_order_relaxed)) {
                                        if (!context.steprequested.exchange(false, std::memory_order_relaxed)) {
                                            break;
                                        }
                                    }
                                }
                                new_epoch_number = context.next_epoch_number;
                                context.next_epoch_number += 1;
                            }
                            if (epoch_view->phase_one_progress.load(std::memory_order_relaxed) != light_samples_per_epoch) {
                                throw "bad";
                            }
                            if (epoch_view->completed_light_paths_count.load(std::memory_order_relaxed) != light_samples_per_epoch) {
                                throw "bad";
                            }
                            if (epoch_view->phase_two_progress.load(std::memory_order_relaxed) != pixels_per_epoch) {
                                throw "bad";
                            }
                            work_performed = true;
                            any_epoch_processed = true;
                            epoch_view->use_count.fetch_add(1, std::memory_order_relaxed);
                            epoch_view->path_node_arena.clear();
                            epoch_view->connector_arena.clear();
                            epoch_view->phase_one_progress.store(0, std::memory_order_relaxed);
                            epoch_view->completed_light_paths_count.store(0, std::memory_order_relaxed);
                            epoch_view->phase_two_progress.store(0, std::memory_order_relaxed);
                            epoch_view->total_resets.fetch_add(1, std::memory_order_relaxed);
                            epoch_view->epoch_number = new_epoch_number;
                            epoch_view->light_triangles_distribution = LightTriangleDistribution(context.world, context.world_surfaces, context.camera);
                        }
                    }
                }
                epoch_view.initialize();
                while (true) {
                    int iq;
                    if (!epoch_view->advance_phase_one_progress(ray_packet_size, iq, light_samples_per_epoch)) {
                        break;
                    }
                    work_performed = true;
                    any_epoch_processed = true;
                    int light_i_begin = iq;
                    int light_i_end = light_i_begin + ray_packet_size;
                    if (light_i_end > light_samples_per_epoch) {
                        light_i_end = light_samples_per_epoch;
                    }
                    for (int i = light_i_begin; i < light_i_end; ++i) {
                        size_t light_triangle_index;
                        epoch_view->light_triangles_distribution.distribution.sample(random, light_triangle_index);
                        float probability_density = epoch_view->light_triangles_distribution.triangle_probability_density[light_triangle_index];
                        World::Triangle const& wtri = context.world.triangles[light_triangle_index];
                        Surface surface = context.triangle_surface(light_triangle_index);
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
                        PathNode& path_node = epoch_view.emplace_path_node(path_node_index);
                        path_node.prev_node_index = path_origin_bit | light_triangle_index;
                        path_node.path_length = 1;
                        path_node.path_index = i;
                        path_node.incoming_value_density = (1.0f / probability_density) * Color{1.0f, 1.0f, 1.0f};
                        path_node.incoming_direction = surface.normal;
                        path_node.position = light_origin;
                        path_node.triangle_index = light_triangle_index;
                        Vec3 light_dir;
                        Color emit_value;
                        surface.sample_emission(random, light_dir, emit_value);
                        path_node.scattered_value_density = (path_node.incoming_value_density * emit_value);
                        if (epoch_view->use_count.fetch_add(1, std::memory_order_acquire) == 0) {
                            throw "bad refcount";
                        }
                        context.current_ray_count.fetch_add(1, std::memory_order_relaxed);
                        epoch_view->total_phase_one_rays_sent.fetch_add(1, std::memory_order_relaxed);
                        inserter.schedule(
                            wtri.zone_id,
                            light_origin,
                            light_dir,
                            make_ray_id(RayType::LightPath, epoch, path_node_index));
                    }
                    context.increment_additive_weight((float)(light_i_end - light_i_begin) / (float)(width * height));
                }
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
        //World world = load_world("C:\\dev\\csg\\maps\\LightTest\\map.bvh");
        World world = load_world("Data\\map.bvh");
        //World world = load_test_world();
        RayPipeline pipeline(world, params);
        //Random random;
        
        //MainCamera camera = MainCamera::targeted(world, Vec3{-24, 115, -65}, Vec3{-24, -3, -92}, 1.0f);
        //MainCamera camera = MainCamera::targeted(world, Vec3{-24, 115, -65}, Vec3{128, -128, 128}, 0.002f);
        MainCamera camera = MainCamera::targeted(world, Vec3{110, -738, 63}, Vec3{-653, -1641, 18}, 1.0f);
        //MainCamera camera = MainCamera::targeted(world, Vec3{160, -1438, 103}, Vec3{-653, -1641, -128}, 1.0f);
        //MainCamera camera = MainCamera::targeted(world, Vec3{108, -1150, 55}, Vec3{64, -1150, 236}, 0.1f);

        Context context(pauserequested, steprequested, world, pipeline, camera);

        std::shared_ptr<ProcessorControl> this_control_ptr = std::make_shared<ProcessorControl>();
        Scheduler::register_processor(std::make_shared<PhaseOneProcessor>(this_control_ptr, context));
        Scheduler::register_processor(std::make_shared<PhaseTwoProcessor>(this_control_ptr, context));
        Scheduler::register_processor(std::make_shared<HandlerProcessor>(this_control_ptr, context));

        uint64_t start_time = GetTickCount64() + 15000;

        while (!rterminate.load(std::memory_order_relaxed)) {
            using namespace std::chrono_literals;

            std::this_thread::sleep_for(100ms);

            double total_power = 0;

            {
                auto& bits = bitbuf.Back();
                bits.resize(width * height * 4);
                uint8_t* pixels = bits.data();
                for (int iy = 0; iy < height; ++iy) {
                    uint8_t* line = pixels + iy * width * 4;
                    for (int ix = 0; ix < width; ++ix) {
                        uint8_t* pixel = line + ix * 4;
                        Color color = context.color_at(ix, iy);
                        tonemap(color, pixel);
                        total_power += color[0] + color[1] + color[2];
                    }
                }
                bitbuf.Publish();
                InvalidateRgn(hwnd, nullptr, false);
            }

            total_power /= width * height;

            char buf[1024];
            if (pauserequested.load(std::memory_order_relaxed)) {
                start_time = GetTickCount64() + 15000;
                snprintf(
                    buf, sizeof(buf),
                    "rt | %llu frames | packets: %.4llu | power: %.8e | flushing",
                    context.total_frames_completed.load(std::memory_order_relaxed),
                    pipeline.get_total_packet_count(),
                    total_power);
            } else {
                uint64_t counter = context.ray_counter.load(std::memory_order_relaxed);
                uint64_t time = GetTickCount64();
                if (time <= start_time) {
                    context.ray_counter.store(0, std::memory_order_relaxed);
                    snprintf(
                        buf, sizeof(buf),
                        "rt | %llu frames | packets: %.4llu | power: %.8e | warming up (%llus)",
                        context.total_frames_completed.load(std::memory_order_relaxed),
                        pipeline.get_total_packet_count(),
                        total_power,
                        (start_time - time) / 1000);
                } else {
                    snprintf(
                        buf, sizeof(buf),
                        "rt | %llu frames | packets: %.4llu | power: %.8e | Krays per second: %llu",
                        context.total_frames_completed.load(std::memory_order_relaxed),
                        pipeline.get_total_packet_count(),
                        total_power,
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

//#pragma optimize( "", on )
