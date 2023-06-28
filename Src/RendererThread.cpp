#include "RendererThread.h"
#include "ParameterSampler.h"
#include "ForwardSampler.h"
#include "BidirSampler.h"
#include "World.h"
#include "RayPipeline.h"
#include "Threading.h"
//#include <png.h>

static int const width = 1600;
static int const height = 900;
static float const exposure = 70.0f;

static bool IsImportantIndex(int index)
{
    switch (index) {
    case 256:
        return false;
    default:
        return false;
    }
}

static bool IsSavedIndex(int index)
{
    switch (index) {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
    case 256:
    case 512:
        return true;
    default:
        return index % 1024 == 0;
    }
}

static float Clamp(float x)
{
    if (x < 0)
        x = 0;
    if (x > 1)
        x = 1;
    return x;
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

static void Tonemap(FDisp value, uint8_t* pixel)
{
    float va, vb, vc;
    value.unpack(va, vb, vc);
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

struct ForwardRay {
    int x, y;
    Vec3 flux;
    Vec3 emission;

    static ForwardRay make(int x, int y) {
        return ForwardRay{
            x, y,
            Vec3{1.0f, 1.0f, 1.0f},
            Vec3{0.0f, 0.0f, 0.0f},
        };
    }
};

struct Context {
    static constexpr size_t accum_buffer_count = 8;

    std::atomic<bool>& pauserequested;
    World& world;
    RayPipeline& pipeline;
    Arena<ForwardRay> fray_arena;
    std::atomic<uint64_t> ray_counter;
    std::mutex accum_buffer_mutexes[accum_buffer_count];
    std::vector<float> accum_buffers[accum_buffer_count];
};

struct RetiredRaysProcessor: Processor {
    Context& context;
    Random random;

    RetiredRaysProcessor(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
        : Processor(std::move(control_ptr))
        , context(context)
    {
    }

    virtual bool has_pending_work_impl() override {
        return !context.pipeline.completed_packets_empty();
    }

    virtual bool work_impl() override {
        bool work_performed = false;
        std::unordered_map<size_t, RayPipeline::Inserter> per_zone_inserter_map;
        Arena<ForwardRay>::View fray_arena_view(context.fray_arena);
        std::unique_lock<std::mutex> guard_accum_buffer;
        size_t selected_buffer = World::invalid_index;
        while (true) {
            std::unique_ptr<RayPipeline::RayPacket> packet_ptr = context.pipeline.pop_completed_packet();
            if (!packet_ptr) {
                break;
            }
            work_performed = true;
            for (int i = 0; i < packet_ptr->ray_count; ++i) {
                RayPipeline::RayData ray_data = packet_ptr->extract_ray_data(i);
                ForwardRay& fray = fray_arena_view[ray_data.extra_data];
                Vec3 normal{0.0f, 0.0f, 1.0f};
                bool does_transmit = false;
                Vec3 flux_gain = Vec3{0.0f, 0.0f, 0.0f};
                float transmit_offset = 0.0f;
                Vec3 emit = Vec3{0.0f, 0.0f, 0.0f};
                size_t target_zone_id = packet_ptr->zone_id;
                if (ray_data.hit_world_triangle_index != World::invalid_index) {
                    World::Triangle const& tri = context.world.zone_trees[packet_ptr->zone_id].triangles[ray_data.hit_world_triangle_index];
                    World::Surface const& surface = context.world.surfaces[tri.surface_index];
                    normal = context.world.normals[tri.normal_index];
                    if (surface.flags & World::flag_invisible) {
                        //does_transmit = true;
                        //flux_gain = Vec3{1.0f, 1.0f, 1.0f};
                        //target_zone_id = tri.other_zone_index;
                    //} else if (surface.flags & World::flag_masked) {
                    //    does_transmit = true;
                    //    flux_gain = Vec3{1.00f, 0.90f, 0.70f};
                    //    transmit_offset = 0.001f;
                    } else {
                        if (false) {
                            emit = Vec3{0.35f, 0.40f, 0.45f};
                            int i = ray_data.hit_world_triangle_index;
                            emit.x = (i % 100) / 100.0f;
                            emit.y = ((i / 100) % 100) / 100.0f;
                            emit.z = ((i / 10000) % 100) / 100.0f;
                        } else {
                            std::string const& mat_name = context.world.material_names[surface.material_index];
                            if (mat_name == "Fire.FireTexture'XFX.xeolighting'") {
                                emit = Vec3{0.35f, 0.40f, 0.45f};
                            } else if (mat_name == "Fire.FireTexture'UTtech1.Misc.donfire'") {
                                emit = Vec3{6.00f, 5.00f, 1.50f};
                            } else if (mat_name == "Engine.Texture'ShaneChurch.Lampon5'") {
                                emit = Vec3{10.00f, 3.00f, 5.00f};
                            //} else if (mat_name == "Fire.WetTexture'LavaFX.Lava3'") {
                            //    emit = Vec3{1.00f, 0.30f, 0.10f};
                            } else if (mat_name == "Engine.Texture'DecayedS.Ceiling.T-CELING2'") {
                                emit = Vec3{10.00f, 11.00f, 12.00f};
                            } else {
                                Vec3 n = norm(normal);
                                if (fabsf(normal.z) <= 0.2f) {
                                    flux_gain = Vec3{0.60f, 0.03f, 0.03f};
                                } else {
                                    flux_gain = Vec3{0.30f, 0.30f, 0.40f};
                                }
                            }
                        }
                    }
                } else {
                    //emit = Vec3{100.00f, 0.00f, 100.00f};
                }
                fray.emission.x += fray.flux.x * emit.x;
                fray.emission.y += fray.flux.y * emit.y;
                fray.emission.z += fray.flux.z * emit.z;
                float reflect_chance = fminf(1.0f, fmaxf(fmaxf(flux_gain.x, flux_gain.y), flux_gain.z));
                bool is_reflected;
                random.p(reflect_chance, is_reflected);
                if (is_reflected) {
                    fray.flux.x *= (1.0f / reflect_chance) * flux_gain.x;
                    fray.flux.y *= (1.0f / reflect_chance) * flux_gain.y;
                    fray.flux.z *= (1.0f / reflect_chance) * flux_gain.z;
                    Vec3 new_origin = ray_data.origin + (ray_data.max_param + transmit_offset) * ray_data.direction;
                    Vec3 new_direction;
                    if (does_transmit) {
                        new_direction = ray_data.direction;
                    } else {
                        random.lambert(normal, new_direction);
                    }
                    auto inserter_iter = per_zone_inserter_map.find(target_zone_id);
                    if (inserter_iter == per_zone_inserter_map.end()) {
                        inserter_iter = per_zone_inserter_map.emplace(std::make_pair(target_zone_id, context.pipeline.inserter(target_zone_id))).first;
                    }
                    inserter_iter->second.schedule(
                        new_origin,
                        new_direction,
                        ray_data.extra_data);
                } else {
                    while (selected_buffer == World::invalid_index) {
                        for (size_t i = 0; i < Context::accum_buffer_count; ++i) {
                            guard_accum_buffer = std::unique_lock<std::mutex>(context.accum_buffer_mutexes[i], std::defer_lock);
                            if (guard_accum_buffer.try_lock()) {
                                selected_buffer = i;
                                break;
                            }
                        }
                    }
                    float* accum_ptr = context.accum_buffers[selected_buffer].data() + fray.y * width * 4 + fray.x * 4;
                    accum_ptr[0] += fray.emission.x;
                    accum_ptr[1] += fray.emission.y;
                    accum_ptr[2] += fray.emission.z;
                    accum_ptr[3] += 1.0;
                    fray_arena_view.schedule_decrement(ray_data.extra_data);
                }
                context.ray_counter.fetch_add(1, std::memory_order_relaxed);
            }
            context.pipeline.collect_ray_packet_spare(std::move(packet_ptr));
            fray_arena_view.commit_decrements();
        }
        return work_performed;
    }
};

struct NewRaysProcessor: Processor {
    Context& context;
    Random random;
    size_t packet_count_threshold;
    std::mutex self_mutex;
    intptr_t iy;

    NewRaysProcessor(std::shared_ptr<ProcessorControl> control_ptr, Context& context)
        : Processor(std::move(control_ptr))
        , context(context)
    {
        //size_t num = width * height * Scheduler::total_worker_count();
        //size_t den = 20 * pipeline.get_params().ray_buffer_size;
        packet_count_threshold = 100000 / context.pipeline.get_params().ray_buffer_size;
        iy = 0;
    }

    virtual bool has_pending_work_impl() override {
        return context.pipeline.get_total_packet_count() < packet_count_threshold;
    }

    virtual bool work_impl() override {
        std::unique_lock<std::mutex> guard(self_mutex, std::defer_lock);

        if (!guard.try_lock()) {
            return false;
        }

        Vec3 origin{110, -738, 63};
        Vec3 target{-653, -1641, 18};
        float ctan = 1.0f;
        Vec3 forward = norm(target - origin);
        Vec3 up{0, 0, 1};
        Vec3 right = norm(cross(forward, up));
        Vec3 down = norm(cross(forward, right));
        float aspect = (float)width / (float)height;
        float utan = ctan * sqrtf(aspect);
        float vtan = ctan / sqrtf(aspect);
        size_t zone_index = context.world.zone_index_at(origin);

        bool work_performed = false;
        RayPipeline::Inserter inserter = context.pipeline.inserter(zone_index);
        Arena<ForwardRay>::Allocator fray_allocator(context.fray_arena);
        while (true) {
            if (context.pipeline.get_total_packet_count() >= packet_count_threshold) {
                break;
            }
            if (iy == 0 && context.pauserequested.load(std::memory_order_relaxed)) {
                break;
            }
            work_performed = true;
            for (intptr_t ix = 0; ix < width; ++ix) {
                float dx, dy;
                random.triangle(dx);
                random.triangle(dy);
                dx *= 1.5f;
                dy *= 1.5f;
                float cx = 2.0f * (ix + dx + 0.5f) / width - 1.0f;
                float cy = 1.0f - 2.0f * (iy + dy + 0.5f) / height;
                float cu = cx * utan;
                float cv = cy * vtan;
                uint64_t fray_index;
                ForwardRay& fray_data = fray_allocator.alloc(fray_index);
                fray_data = ForwardRay::make(ix, iy);
                inserter.schedule(
                    origin,
                    forward + cu * right + cv * down,
                    fray_index);
            }
            iy = (iy + 1) % height;
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

        //Vec3 origin{110, -738, 63};
        //Vec3 target{-653, -1641, 18};
        //float ctan = 1.0f;
        //Vec3 forward = norm(target - origin);
        //Vec3 up{0, 0, 1};
        //Vec3 right = norm(cross(forward, up));
        //Vec3 down = norm(cross(forward, right));
        //float aspect = (float)width / (float)height;
        //float utan = ctan * sqrtf(aspect);
        //float vtan = ctan / sqrtf(aspect);
        //// 35, 61, 98, 165
        //// 99, 156
        //size_t zone_index = world.zone_index_at(origin);

        Context context{pauserequested, world, pipeline};
        context.ray_counter = 0;
        for (size_t i = 0; i < Context::accum_buffer_count; ++i) {
            context.accum_buffers[i].resize(width * height * 4, 0.0f);
        }

        std::shared_ptr<ProcessorControl> this_control_ptr = std::make_shared<ProcessorControl>();
        for (int i = 0; i < 2; ++i) {
            Scheduler::register_processor(std::make_shared<NewRaysProcessor>(this_control_ptr, context));
        }
        Scheduler::register_processor(std::make_shared<RetiredRaysProcessor>(this_control_ptr, context));

        std::vector<double> total_accum_buffer(width * height * 4);

        auto to_pixel_buffer = [&](uint8_t* pixels) {
            for (intptr_t iy = 0; iy < height; ++iy) {
                uint8_t* line = pixels + iy * width * 4;
                for (intptr_t ix = 0; ix < width; ++ix) {
                    uint8_t* pixel = line + ix * 4;
                    double* accum_ptr = total_accum_buffer.data() + iy * width * 4 + ix * 4;
                    float r = 0, g = 0, b = 0;
                    if (accum_ptr[3] != 0) {
                        r = (float)(accum_ptr[0] / accum_ptr[3]);
                        g = (float)(accum_ptr[1] / accum_ptr[3]);
                        b = (float)(accum_ptr[2] / accum_ptr[3]);
                    }
                    Tonemap(FDisp(r, g, b), pixel);
                }
            }
        };

        uint64_t start_time = GetTickCount64() + 15000;
        size_t current_accum_index = 0;

        while (!rterminate.load(std::memory_order_relaxed)) {
            using namespace std::chrono_literals;

            std::this_thread::sleep_for(100ms);

            {
                std::lock_guard<std::mutex> guard(context.accum_buffer_mutexes[current_accum_index]);
                for (size_t k = 0; k < total_accum_buffer.size(); ++k) {
                    total_accum_buffer[k] += context.accum_buffers[current_accum_index][k];
                    context.accum_buffers[current_accum_index][k] = 0.0f;
                }
            }
            current_accum_index = (current_accum_index + 1) % Context::accum_buffer_count;

            {
                auto& bits = bitbuf.Back();
                bits.resize(width * height * 4);
                to_pixel_buffer(bits.data());
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
