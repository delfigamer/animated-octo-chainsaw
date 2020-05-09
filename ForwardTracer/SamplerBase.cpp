#define _CRT_SECURE_NO_WARNINGS

#include "SamplerBase.h"
#include "header.h"
#include <cstdio>

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

void SamplerBase::RecordToFrame(int x, int y, FDisp value)
{
    if (x >= 0 && x < width && y >= 0 && y < height) {
        float a, b, c;
        value.unpack(a, b, c);
        frames[currentframe][y * width + x][0] += a;
        frames[currentframe][y * width + x][1] += b;
        frames[currentframe][y * width + x][2] += c;
    }
}

SamplerBase::SamplerBase(int width, int height)
    : width(width)
    , height(height)
    , perf{ traceperf.samples, sampleperf.samples }
{
    for (int p = 0; p < FrameCount; ++p) {
        frames[p].resize(width * height, { 0, 0, 0 });
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
    for (int iy = height - 1; iy >= 0; --iy) {
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
        for (int i = 0; i < width * height; ++i) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0;
                double sumsqr = 0;
                for (int p = 0; p < FrameCount; ++p) {
                    double value = frames[p][i][c] / denominator;
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
    {
        FILE* f = fopen("D:\\rt\\export.txt", "w");
        int rwidth = width / 100;
        int rheight = height / 100;
        for (int ry = rheight - 1; ry >= 0; --ry) {
            for (int rx = 0; rx < rwidth; ++rx) {
                int minx = width * rx / rwidth;
                int maxx = width * (rx + 1) / rwidth;
                int miny = height * ry / rheight;
                int maxy = height * (ry + 1) / rheight;
                double accum = 0;
                for (int p = 0; p < FrameCount; ++p) {
                    for (int iy = miny; iy < maxy; ++iy) {
                        for (int ix = minx; ix < maxx; ++ix) {
                            FDisp d = FDisp{
                                (float)frames[p][iy * width + ix][0],
                                (float)frames[p][iy * width + ix][1],
                                (float)frames[p][iy * width + ix][2] };
                            accum += luma(d);
                        }
                    }
                }
                accum /= denominator * (maxx - minx) * (maxy - miny);
                fprintf(f, " %10.6lf", accum);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
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
}
