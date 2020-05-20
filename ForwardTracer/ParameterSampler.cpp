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
        FPoint{ -130, -140, 150 },
        FPoint{ -10, 0, 120 },
        1.0f * sqrtf(aspect), 1.0f / sqrtf(aspect));
}

ParameterSampler::~ParameterSampler()
{
}

void ParameterSampler::IteratePixel(int ix, int iy)
{
    float passrate = 1.0f / 128;
    if (!RandomBool(passrate)) {
        return;
    }
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
    if (Trace(tr)) {
        FPoint chit = camera.mcw * tr.hit;
        float chx, chy, chz;
        chit.unpack(chx, chy, chz);
        float a, b, c;
        tr.hitlocal.unpack(a, b, c);
        c = 1 - a - b;
        FDisp kav = FDisp{ scene[tr.face].mtw.xdual() };
        FDisp kbv = FDisp{ scene[tr.face].mtw.ydual() };
        float p = 100;
        float kea = expf(-p * a);
        float keb = expf(-p * b);
        float kec = expf(-p * c);
        float k = (1 - kea) * (1 - keb) * (1 - kec);
        float kda = p * kea * (1 - keb) * (1 - kec);
        float kdb = p * (1 - kea) * keb * (1 - kec);
        float kdc = p * (1 - kea) * (1 - keb) * kec;
        FDisp kgrad = (kda - kdc) * kav + (kdb - kdc) * kbv;
        FDisp d = tr.hit - camera.mwc.origin();
        FDisp norm = scene[tr.face].mwt.zunit();
        float fluxa, fluxb, fluxc;
        float rua, rub, ruc;
        float rva, rvb, rvc;
        {
            float flux = a;
            FDisp fluxgrad = kav;
            FDisp finalgrad = flux * kgrad + k * fluxgrad;
            FDisp radialgrad = finalgrad - (dot(d, finalgrad) / dot(d, norm)) * norm;
            FDisp radialcam = camera.mcw * radialgrad;
            float rdx, rdy, rdz;
            radialcam.unpack(rdx, rdy, rdz);
            float ru = rdx * chz;
            float rv = rdy * chz;
            fluxa = flux; rua = ru; rva = rv;
        }
        {
            float flux = b;
            FDisp fluxgrad = kbv;
            FDisp finalgrad = flux * kgrad + k * fluxgrad;
            FDisp radialgrad = finalgrad - (dot(d, finalgrad) / dot(d, norm)) * norm;
            FDisp radialcam = camera.mcw * radialgrad;
            float rdx, rdy, rdz;
            radialcam.unpack(rdx, rdy, rdz);
            float ru = rdx * chz;
            float rv = rdy * chz;
            fluxb = flux; rub = ru; rvb = rv;
        }
        {
            float flux = c;
            FDisp fluxgrad = - kav - kbv;
            FDisp finalgrad = flux * kgrad + k * fluxgrad;
            FDisp radialgrad = finalgrad - (dot(d, finalgrad) / dot(d, norm)) * norm;
            FDisp radialcam = camera.mcw * radialgrad;
            float rdx, rdy, rdz;
            radialcam.unpack(rdx, rdy, rdz);
            float ru = rdx * chz;
            float rv = rdy * chz;
            fluxc = flux; ruc = ru; rvc = rv;
        }
        RecordToFrame(
            ix + dx, iy + dy,
            0.1f / passrate * FDisp{ fluxa, fluxb, fluxc },
            0.1f / passrate * (1 - k) * FDisp{ fluxa, fluxb, fluxc },
            0.2f / passrate * camera.utan * FDisp{ rua, rub, ruc },
            0.2f / passrate * camera.vtan * FDisp{ rva, rvb, rvc });
    }
}
