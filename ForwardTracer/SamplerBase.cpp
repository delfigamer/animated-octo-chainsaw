#define _CRT_SECURE_NO_WARNINGS

#include "SamplerBase.h"
#include "header.h"
#include <cstdio>
#include <cmath>

static float DeltaFactor(int a, int b)
{
    return a == b ? 1.0f : 0.0f;
}

static float DctFactor(int a, int b)
{
    static float table[8][8] = {
        { +0.35355339f, +0.35355339f, +0.35355339f, +0.35355339f, +0.35355339f, +0.35355339f, +0.35355339f, +0.35355339f },
        { +0.49039264f, +0.41573480f, +0.27778511f, +0.09754516f, -0.09754516f, -0.27778511f, -0.41573480f, -0.49039264f },
        { +0.46193976f, +0.19134171f, -0.19134171f, -0.46193976f, -0.46193976f, -0.19134171f, +0.19134171f, +0.46193976f },
        { +0.41573480f, -0.09754516f, -0.49039264f, -0.27778511f, +0.27778511f, +0.49039264f, +0.09754516f, -0.41573480f },
        { +0.35355339f, -0.35355339f, -0.35355339f, +0.35355339f, +0.35355339f, -0.35355339f, -0.35355339f, +0.35355339f },
        { +0.27778511f, -0.49039264f, +0.09754516f, +0.41573480f, -0.41573480f, -0.09754516f, +0.49039264f, -0.27778511f },
        { +0.19134171f, -0.46193976f, +0.46193976f, -0.19134171f, -0.19134171f, +0.46193976f, -0.46193976f, +0.19134171f },
        { +0.09754516f, -0.27778511f, +0.41573480f, -0.49039264f, +0.49039264f, -0.41573480f, +0.27778511f, -0.09754516f } };
    if (a >= 0 && a < 8 && b >= 0 && b < 8) {
        return table[a][b];
    } else {
        return 0;
    }
}

static float HaarFactor(int a, int b)
{
    constexpr float q1 = 0.3535534f;
    constexpr float q2 = 0.5f;
    constexpr float q3 = 0.7071068f;
    static float table[8][8] = {
        { q1, q1, q1, q1, q1, q1, q1, q1 },
        { q1, q1, q1, q1, -q1, -q1, -q1, -q1 },
        { q2, q2, -q2, -q2, 0, 0, 0, 0 },
        { 0, 0, 0, 0, q2, q2, -q2, -q2 },
        { q3, -q3, 0, 0, 0, 0, 0, 0 },
        { 0, 0, q3, -q3, 0, 0, 0, 0 },
        { 0, 0, 0, 0, q3, -q3, 0, 0 },
        { 0, 0, 0, 0, 0, 0, q3, -q3 } };
    if (a >= 0 && a < 8 && b >= 0 && b < 8) {
        return table[a][b];
    } else {
        return 0;
    }
}

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

float SamplerBase::SupportKernel(float x)
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

float SamplerBase::SupportIntegral(float x)
{
    if (x < -1) {
        return 0;
    } else if (x < 0) {
        return (-0.5f * x + 1.0f) * x + 0.5f;
    } else if (x < 1) {
        return (0.5f * x + 1.0f) * x + 0.5f;
    } else {
        return 1;
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
    int minxb = Clip(Floor(0.125f * x), 0, blockwidth);
    int maxxb = Clip(Ceil(0.125f * (x + 1)), 0, blockwidth);
    int minyb = Clip(Floor(0.125f * y), 0, blockheight);
    int maxyb = Clip(Ceil(0.125f * (y + 1)), 0, blockheight);
    float invwidth = 1.0f / width;
    float invheight = 1.0f / height;
    for (int yb = minyb; yb < maxyb; ++yb) {
        for (int xb = minxb; xb < maxxb; ++xb) {
            float rx = x - xb * 8;
            float ry = y - yb * 8;
            FrameBlock& block = frames[currentframe][yb * blockwidth + xb];
            for (int yf = 0; yf < 8; ++yf) {
                for (int xf = 0; xf < 8; ++xf) {
                    float vfactor = 0;
                    float dxfactor = 0;
                    float dyfactor = 0;
                    for (int yt = 0; yt < 8; ++yt) {
                        for (int xt = 0; xt < 8; ++xt) {
                            float xform = HaarFactor(xf, xt) * HaarFactor(yf, yt);
                            float dx = rx - xt;
                            float dy = ry - yt;
                            vfactor += xform * SupportKernel(dx) * SupportKernel(dy);
                            dxfactor += xform * SupportIntegral(dx) * SupportKernel(dy);
                            dyfactor += xform * SupportKernel(dx) * SupportIntegral(dy);
                        }
                    }
                    FDisp value;
                    if (xf == 0) {
                        if (yf == 0) {
                            value = vfactor * vfull;
                        } else {
                            value = vfactor * vpart + invheight * dyfactor * dvdy;
                        }
                    } else {
                        if (yf == 0) {
                            value = vfactor * vpart - invwidth * dxfactor * dvdx;
                        } else {
                            __m128 xweight = _mm_add_ps(_mm_mul_ps(dvdx.m, dvdx.m), _mm_set_ps1(0.01f));
                            __m128 yweight = _mm_add_ps(_mm_mul_ps(dvdy.m, dvdy.m), _mm_set_ps1(0.01f));
                            __m128 den = _mm_rcp_ps(_mm_add_ps(xweight, yweight));
                            xweight = _mm_mul_ps(xweight, den);
                            yweight = _mm_mul_ps(yweight, den);
                            FDisp dvdxw = FDisp{ _mm_mul_ps(xweight, dvdx.m) };
                            FDisp dvdyw = FDisp{ _mm_mul_ps(yweight, dvdy.m) };
                            value = vfactor * vpart - invwidth * dxfactor * dvdxw + invheight * dyfactor * dvdyw;
                        }
                    }
                    float a, b, c;
                    value.unpack(a, b, c);
                    block[yf * 8 + xf][0] += a;
                    block[yf * 8 + xf][1] += b;
                    block[yf * 8 + xf][2] += c;
                }
            }
        }
    }
}

SamplerBase::SamplerBase(int width, int height)
    : width(width)
    , height(height)
    , perf{ traceperf.samples, sampleperf.samples }
{
    blockwidth = (width + 7) / 8;
    blockheight = (height + 7) / 8;
    for (int p = 0; p < FrameCount; ++p) {
        frames[p].resize(blockwidth * blockheight, FrameBlock{});
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
        int xb = x / 8;
        int yb = y / 8;
        int xt = x - xb * 8;
        int yt = y - yb * 8;
        for (int p = 0; p < FrameCount; ++p) {
            FrameBlock& block = frames[p][yb * blockwidth + xb];
            for (int xf = 0; xf < 8; ++xf) {
                for (int yf = 0; yf < 8; ++yf) {
                    float factor = HaarFactor(xf, xt) * HaarFactor(yf, yt);
                    if (xf + yf != 1) {
                        //continue;
                    }
                    a += factor * block[yf * 8 + xf][0];
                    b += factor * block[yf * 8 + xf][1];
                    c += factor * block[yf * 8 + xf][2];
                }
            }
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
        for (int b = 0; b < blockwidth * blockheight; ++b) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0;
                double sumsqr = 0;
                for (int p = 0; p < FrameCount; ++p) {
                    for (int f = 0; f < 64; ++f) {
                        double value = frames[p][b][f][c] / denominator;
                        sum += value;
                        sumsqr += value * value;
                    }
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
