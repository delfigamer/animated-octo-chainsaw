#pragma once

#include "FLin.h"
#include "Geometry.h"
#include <vector>
#include <map>
#include <array>
#include <random>

class SamplerBase
{
protected:
    static constexpr int FrameCount = 4;

    struct PerfCounter
    {
        std::map<int64_t, int> samples;

        void SampleStart(int64_t& tmp);
        void SampleEnd(int64_t tmp);
        int64_t Time();
    };

    using FramePixel = std::array<double, 3>;

    struct Kernel
    {
        float width;

        Kernel(float width);
        void Sample(float q, float& r, float& d, float& dl);
    };

public:
    struct PerfInfo
    {
        std::map<int64_t, int>& traceTimes;
        std::map<int64_t, int>& sampleTimes;
        int64_t traceTime;
        int64_t sampleTime;
        float error;
    };

protected:
    int width;
    int height;
    Geometry scene;
    Kernel pixelkernel;
    Kernel extendkernel;
    std::mt19937 rand;
    std::array<std::vector<FramePixel>, FrameCount> frames;
    std::array<std::vector<double>, FrameCount> frameden;
    int currentframe;
    double denominator;
    PerfCounter traceperf;
    PerfCounter sampleperf;
    PerfInfo perf;

    void GenerateUniform(float& q);
    void GenerateTriangle(float& u);
    void GenerateCircle(float& u, float& v);
    void GenerateLambertian(FDisp const& n, FDisp& d);
    bool RandomBool(float beta);
    bool Trace(TraceRequest& tr);
    bool TestVisible(FPoint from, FPoint to);
    void RecordToFrame(float x, float y, float wfull, float wpart, FDisp vfull, FDisp vpart, FDisp dvdx, FDisp dvdy);

public:
    SamplerBase(int width, int height);
    ~SamplerBase();

    virtual void IteratePixel(int ix, int iy) = 0;
    void Iterate();
    FDisp GetValue(int x, int y);
    PerfInfo const& GetPerfInfo();
    void Export();
};
