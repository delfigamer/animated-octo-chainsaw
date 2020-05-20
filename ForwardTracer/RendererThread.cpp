#include "RendererThread.h"
#include "ParameterSampler.h"
#include "ForwardSampler.h"
#include "BidirSampler.h"
#include <png.h>

static int const width = 800;
static int const height = 600;
static float const exposure = 10.0f;

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
        return true;
    default:
        return true;
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

static void Tonemap(FDisp value, uint8_t* pixel)
{
    float ta, tb, tc;
    if (true) {
        float va, vb, vc;
        value.unpack(va, vb, vc);
        ta = 1.0f - expf(-va);
        tb = 1.0f - expf(-vb);
        tc = 1.0f - expf(-vc);
    } else {
        float l = luma(value);
        ta = 0;
        tb = 0;
        tc = 0;
        if (l > 0) {
            tc = 1.0f - expf(-l);
        } else {
            ta = 1.0f - expf(l);
        }
    }
    ta = fminf(fmaxf(ta, 0), 1);
    tb = fminf(fmaxf(tb, 0), 1);
    tc = fminf(fmaxf(tc, 0), 1);
    int ba = (int)(sqrtf(ta) * 255.0f);
    int bb = (int)(sqrtf(tb) * 255.0f);
    int bc = (int)(sqrtf(tc) * 255.0f);
    pixel[0] = bc;
    pixel[1] = bb;
    pixel[2] = ba;
    pixel[3] = 255;
}

void RendererThread::ThreadFunc()
{
    //std::unique_ptr<SamplerBase> pintegrator{ new BidirSampler(width, height) };
    //std::unique_ptr<SamplerBase> pintegrator{ new ForwardSampler(width, height) };
    std::unique_ptr<SamplerBase> pintegrator{ new ParameterSampler(width, height) };
    int64_t time = GetTickCount64();
    int iterindex = 0;
    while (!rterminate.load(std::memory_order_relaxed)) {
        if (exportrequested.load(std::memory_order_relaxed)) {
            pintegrator->Export();
            exportrequested.store(false, std::memory_order_relaxed);
        }
        if (pauserequested.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
        } else {
            iterindex += 1;
            pintegrator->Iterate();
            auto& bits = bitbuf.Back();
            bits.resize(width * height * 4);
            uint8_t* pixels = bits.data();
            for (int iy = 0; iy < height; ++iy) {
                uint8_t* line = pixels + iy * width * 4;
                for (int ix = 0; ix < width; ++ix) {
                    uint8_t* pixel = line + ix * 4;
                    FDisp value = exposure * pintegrator->GetValue(ix, iy);
                    Tonemap(value, pixel);
                }
            }
            auto& perf = pintegrator->GetPerfInfo();
            char buf[1024];
            if (IsSavedIndex(iterindex)) {
                png_image pi = {};
                pi.opaque = nullptr;
                pi.version = PNG_IMAGE_VERSION;
                pi.width = width;
                pi.height = height;
                pi.format = PNG_FORMAT_BGRA;
                pi.flags = 0;
                pi.colormap_entries = 0;
                snprintf(
                    buf, sizeof(buf),
                    "D:\\rt\\output\\%.5d.png",
                    iterindex);
                png_image_write_to_file(&pi, buf, false, bits.data(), -width * 4, nullptr);
            }
            bitbuf.Publish();
            InvalidateRgn(hwnd, nullptr, false);
            snprintf(
                buf, sizeof(buf),
                "rt | %i | %lli | %8.6f",
                iterindex, GetTickCount64() - time, perf.error);
            SendMessageTimeoutA(
                hwnd, WM_SETTEXT, (WPARAM)nullptr, (LPARAM)buf,
                SMTO_NORMAL, 100, nullptr);
            if (IsImportantIndex(iterindex) && !pauserequested.load(std::memory_order_relaxed)) {
                SetForegroundWindow(hwnd);
                pauserequested.store(true, std::memory_order_relaxed);
            }
        }
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
