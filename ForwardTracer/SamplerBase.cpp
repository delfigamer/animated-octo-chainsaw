#define _CRT_SECURE_NO_WARNINGS

#include "SamplerBase.h"
#include "header.h"
#include <cstdio>
#include <cmath>

void SamplerBase::PerfCounter::SampleStart(int64_t& tmp)
{
    tmp = (int64_t)__rdtsc();
}

void SamplerBase::PerfCounter::SampleEnd(int64_t tmp)
{
    int64_t delta = (int64_t)__rdtsc() - tmp;
    auto it = samples.try_emplace(delta, 0).first;
    it->second += 1;
}

int64_t SamplerBase::PerfCounter::Time()
{
    int64_t value = 0;
    int count = 0;
    for (auto const& kv : samples) {
        value += kv.second * kv.first;
        count += kv.second;
    }
    if (count > 0) {
        return value / count;
    } else {
        return 0;
    }
}

void SamplerBase::GenerateUniform(float& q)
{
    std::uniform_real_distribution<float> dist;
    q = dist(rand);
}

void SamplerBase::GenerateTriangle(float& u)
{
    float q1, q2;
    GenerateUniform(q1);
    GenerateUniform(q2);
    u = q1 - q2;
}

void SamplerBase::GenerateCircle(float& u, float& v)
{
    while (true) {
        GenerateUniform(u);
        u = u * 2.0f - 1.0f;
        GenerateUniform(v);
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

void SamplerBase::GenerateLambertian(FDisp const& n, FDisp& d)
{
    FDisp a{ 1, 0, 0 };
    float nx, ny, nz;
    n.unpack(nx, ny, nz);
    if (nx > 0.8f || nx < -0.8f) {
        a = FDisp{ 0, 1, 0 };
    }
    FDisp b = norm(cross(a, n));
    FDisp t = norm(cross(b, n));
    float q;
    GenerateUniform(q);
    float z = sqrtf(q);
    float r = sqrtf(1.0f - q);
    float u;
    float v;
    GenerateCircle(u, v);
    u *= r;
    v *= r;
    d = z * n + u * b + v * t;
}

bool SamplerBase::RandomBool(float beta)
{
    float q;
    GenerateUniform(q);
    return q < beta;
}

bool SamplerBase::Trace(TraceRequest& tr)
{
    int64_t time;
    traceperf.SampleStart(time);
    bool r = scene.Trace(tr);
    traceperf.SampleEnd(time);
    return r;
}

bool SamplerBase::TestVisible(FPoint from, FPoint to)
{
    int64_t time;
    traceperf.SampleStart(time);
    bool r = scene.Test(from, to - from);
    traceperf.SampleEnd(time);
    return r;
}

float SamplerBase::KernelP(float x)
{
    if (x < -1) {
        return 0;
    } else if (x < 0) {
        return 1 + x;
    } else if (x < 1) {
        return 1 - x;
    } else {
        return 0;
    }
}

float SamplerBase::KernelA(float x)
{
    if (x < -5) {
        return 0;
    } else if (x < -4) {
        return (1.0f / 9) * (5 + x);
    } else if (x < 4) {
        return (1.0f / 9);
    } else if (x < 5) {
        return (1.0f / 9) * (5 - x);
    } else {
        return 0;
    }
}

float SamplerBase::KernelB(float x)
{
    if (x < -5) {
        return 0;
    } else if (x < -4) {
        float t = 5 + x;
        return - (1.0f / 18) * t * t;
    } else if (x < -1) {
        return - (1.0f / 9) * (4.5f + x);
    } else if (x < 0) {
        return (1.0f / 2) * x * (16 / 9.0f + x);
    } else if (x < 1) {
        return - (1.0f / 2) * x * (- 16 / 9.0f + x);
    } else if (x < 4) {
        return (1.0f / 9) * (4.5f - x);
    } else if (x < 5) {
        float t = -5 + x;
        return (1.0f / 18) * t * t;
    } else {
        return 0;
    }
}

static int Clip(int x, int min, int max)
{
    if (x < min) {
        return min;
    } else if (x > max) {
        return max;
    } else {
        return x;
    }
}

static int Floor(float x)
{
    return (int)floorf(x);
}

static int Ceil(float x)
{
    return (int)ceilf(x);
}

void SamplerBase::RecordToFrame(float x, float y, FDisp vfull, FDisp vpart, FDisp dvdx, FDisp dvdy)
{
    int minx = Clip(Floor(x - 5), 0, width);
    int maxx = Clip(Ceil(x + 6), 0, width);
    int miny = Clip(Floor(y - 5), 0, height);
    int maxy = Clip(Ceil(y + 6), 0, height);
    float invwidth = 1.0f / width;
    float invheight = 1.0f / height;
    for (int ty = miny; ty < maxy; ++ty) {
        for (int tx = minx; tx < maxx; ++tx) {
            float dx = x - tx;
            float dy = y - ty;
            float vffactor = KernelP(dx) * KernelP(dy);
            float vpfactor = KernelA(dx) * KernelA(dy);
            float dxfactor = KernelB(dx) * KernelA(dy);
            float dyfactor = KernelA(dx) * KernelB(dy);
            __m128 xweight = _mm_add_ps(_mm_mul_ps(dvdx.m, dvdx.m), _mm_set_ps1(0.01f));
            __m128 yweight = _mm_add_ps(_mm_mul_ps(dvdy.m, dvdy.m), _mm_set_ps1(0.01f));
            __m128 den = _mm_rcp_ps(_mm_add_ps(xweight, yweight));
            xweight = _mm_mul_ps(xweight, den);
            yweight = _mm_mul_ps(yweight, den);
            FDisp dvdxw = FDisp{ _mm_mul_ps(xweight, dvdx.m) };
            FDisp dvdyw = FDisp{ _mm_mul_ps(yweight, dvdy.m) };
            FDisp value = vffactor * vpart + vpfactor * (vfull - vpart) - invwidth * dxfactor * dvdxw + invheight * dyfactor * dvdyw;
            float a, b, c;
            value.unpack(a, b, c);
            frames[currentframe][ty * width + tx][0] += a;
            frames[currentframe][ty * width + tx][1] += b;
            frames[currentframe][ty * width + tx][2] += c;
        }
    }
}

SamplerBase::SamplerBase(int width, int height)
    : width(width)
    , height(height)
    , perf{ traceperf.samples, sampleperf.samples }
{
    for (int p = 0; p < FrameCount; ++p) {
        frames[p].resize(width * height, FramePixel{});
    }
    denominator = 0;
    currentframe = 0;
}

SamplerBase::~SamplerBase()
{
}

void SamplerBase::Iterate()
{
    denominator += 1;
    for (int iy = 0; iy < height; ++iy) {
        for (int ix = 0; ix < width; ++ix) {
            int64_t time;
            sampleperf.SampleStart(time);
            IteratePixel(ix, iy);
            sampleperf.SampleEnd(time);
        }
    }
    currentframe = (currentframe + 1) % FrameCount;
}

FDisp SamplerBase::GetValue(int x, int y)
{
    if (x >= 0 && x < width && y >= 0 && y < height) {
        double a = 0;
        double b = 0;
        double c = 0;
        for (int p = 0; p < FrameCount; ++p) {
            a += frames[p][y * width + x][0];
            b += frames[p][y * width + x][1];
            c += frames[p][y * width + x][2];
        }
        a /= denominator;
        b /= denominator;
        c /= denominator;
        return FDisp{ (float)a, (float)b, (float)c };
    } else {
        return FDisp{ 0, 0, 0 };
    }
}

SamplerBase::PerfInfo const& SamplerBase::GetPerfInfo()
{
    perf.traceTime = traceperf.Time();
    perf.sampleTime = sampleperf.Time();
    if (denominator > 0) {
        double errorsqr = 0;
        for (int xy = 0; xy < width * height; ++xy) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0;
                double sumsqr = 0;
                for (int p = 0; p < FrameCount; ++p) {
                    double value = frames[p][xy][c] / denominator;
                    sum += value;
                    sumsqr += value * value;
                }
                double avg = sum;
                double avgsqr = sumsqr * FrameCount;
                double var = avgsqr - avg * avg;
                errorsqr += var;
            }
        }
        double rms = sqrt(errorsqr / (width * height));
        perf.error = (float)rms;
    } else {
        perf.error = 0;
    }
    return perf;
}

void SamplerBase::Export()
{
    /*
    {
        FILE* f = fopen("D:\\rt\\export.txt", "w");
        int rwidth = width / 1;
        int rheight = height / 1;
        for (int ry = rheight - 1; ry >= 0; --ry) {
            for (int rx = 0; rx < rwidth; ++rx) {
                int minx = width * rx / rwidth;
                int maxx = width * (rx + 1) / rwidth;
                int miny = height * ry / rheight;
                int maxy = height * (ry + 1) / rheight;
                double accum = 0;
                for (int iy = miny; iy < maxy; ++iy) {
                    for (int ix = minx; ix < maxx; ++ix) {
                        FDisp d = GetValue(ix, iy);
                        accum += luma(d);
                    }
                }
                accum /= denominator * (maxx - minx) * (maxy - miny);
                fprintf(f, " %10.6lf", accum);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
    */
    /*
    {
        FILE* f = fopen("D:\\rt\\export.csv", "w");
        for (int col = 0; col < 3; ++col) {
            for (int ry = height - 1; ry >= 0; --ry) {
                for (int rx = 0; rx < width; ++rx) {
                    double accum = 0;
                    for (int p = 0; p < FrameCount; ++p) {
                        accum += frames[0][ry * width + rx][0];
                    }
                    fprintf(f, ", %.6lg", accum / denominator);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
    */
    /*
    {
        FILE* f = fopen("D:\\rt\\traceperf.csv", "w");
        for (auto const& kv : traceperf.samples) {
            fprintf(f, "%lli, %i\n", kv.first, kv.second);
        }
        fclose(f);
    }
    {
        FILE* f = fopen("D:\\rt\\sampleperf.csv", "w");
        for (auto const& kv : sampleperf.samples) {
            fprintf(f, "%lli, %i\n", kv.first, kv.second);
        }
        fclose(f);
    }
    */
}
