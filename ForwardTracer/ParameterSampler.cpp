#include "ParameterSampler.h"
#include "header.h"
#include <cmath>

ParameterSampler::ParameterSampler(int width, int height)
    : SamplerBase(width, height)
{
    float aspect = (float)width / height;
    /*scene.LoadFromT3D("D:\\Program Files (x86)\\Unreal Tournament GOTY\\Maps\\CTF-Coret-FlagRoom.t3d");
    camera = Camera::Targeted(
        FPoint{ -160, 228, 410 },
        FPoint{ 288, -384, 240 },
        1.0f * sqrtf(aspect), 1.0f / sqrtf(aspect));*/
    scene.LoadFromT3D("D:\\Program Files (x86)\\Unreal Tournament GOTY\\Maps\\Box.t3d");
    camera = Camera::Targeted(
        FPoint{ -170, 140, 230 },
        FPoint{ 0, -20, 50 },
        1.0f * sqrtf(aspect), 1.0f / sqrtf(aspect));
}

ParameterSampler::~ParameterSampler()
{
}

void ParameterSampler::IteratePixel(int ix, int iy)
{
    float dx;
    float dy;
    GenerateTriangle(dx);
    GenerateTriangle(dy);
    float cx = 2.0f * (ix + dx + 0.5f) / width - 1.0f;
    float cy = 1.0f - 2.0f * (iy + dy + 0.5f) / height;
    float cu = cx * camera.utan;
    float cv = cy * camera.vtan;
    TraceRequest tr;
    tr.origin = camera.mwc.origin();
    tr.dir = norm(camera.mwc * FDisp{ cu, cv, 1.0f });
    FDisp current = FDisp{ 0, 0, 0 };
    float factor = 1;
    while (Trace(tr)) {
        float texu = dot(tr.hit, scene[tr.face].texu);
        float texv = dot(tr.hit, scene[tr.face].texv);
        // FDisp tc = scene.SampleColor(scene[tr.face].diffusetex, texu, texv);
        FDisp tc;
        if (tr.side == 0) {
            tc = FDisp{ 0, 1, 0 };
        } else if (tr.side == 1) {
            tc = FDisp{ 0, 0, 1 };
        } else {
            tc = FDisp{ 1, 0, 0 };
        }
        current += tc * factor;
        factor *= 0.5f;
        tr.origin = tr.hit;
    }
    RecordToFrame(ix, iy, 0.2f * current);
}
