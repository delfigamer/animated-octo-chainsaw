#include "ForwardSampler.h"
#include "Spectrum.h"
#include <algorithm>
#include <cmath>

ForwardSampler::ForwardSampler(int width, int height)
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

ForwardSampler::~ForwardSampler()
{
}

void ForwardSampler::IteratePixel(int ix, int iy)
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
    Spectrum current = Spectrum::One();
    int pathlen = 1;
    while (Trace(tr)) {
        pathlen += 1;
        float tu = dot(tr.hit, scene[tr.face].texu);
        float tv = dot(tr.hit, scene[tr.face].texv);
        FaceMaterial const& fmat = scene[tr.face].mat;
        if (true || pathlen == 3) {
            FDisp emitcolor = scene.SampleColor(fmat.emittex, tu, tv) * fmat.emitgain;
            Spectrum emitspec = Spectrum::EmissionSpectrum(emitcolor);
            modulate(emitspec, current);
            float r = dot(emitspec, Spectrum::ResponseSpectrumR());
            float g = dot(emitspec, Spectrum::ResponseSpectrumG());
            float b = dot(emitspec, Spectrum::ResponseSpectrumB());
            RecordToFrame(ix, iy, FDisp{ r, g, b });
            //break;
        }
        FDisp normal = scene[tr.face].mwt.zunit();
        FDisp albedocolor = scene.SampleColor(fmat.albedotex, tu, tv);
        float transparency = luma(scene.SampleColor(fmat.transparencytex, tu, tv));
        float roughness = luma(scene.SampleColor(fmat.roughnesstex, tu, tv));
        float refrindex = fmat.refrindex;
        if (tr.side == -1) {
            refrindex = 1 / refrindex;
            normal = -normal;
            if (fmat.transmittancespec >= 0) {
                Spectrum transmitspec = Spectrum::TableSpectrum(fmat.transmittancespec);
                float power = sqrtf(dot(tr.hit - tr.origin, tr.hit - tr.origin)) / fmat.transmittancelength;
                modulate(transmitspec, power);
                Spectrum filter = transmitspec.exp2();
                float passprob = 1;// luma(filter);
                if (!RandomBool(passprob)) {
                    break;
                }
                modulate(current, filter, 1 / passprob);
            }
        }
        if (RandomBool(transparency)) {
            bool refracted = false;
            FDisp rdir;
            float cosi = -dot(tr.dir, normal);
            float fresnel = 1;
            float t = refrindex * refrindex + cosi * cosi - 1;
            if (t > 0) {
                float g = sqrtf(t);
                float t1 = (g - cosi) / (g + cosi);
                float t2 = (cosi * (g + cosi) - 1) / (cosi * (g - cosi) + 1);
                fresnel = 0.5f * t1 * t1 * (1 + t2 * t2);
                if (RandomBool(1 - fresnel)) {
                    float hy = g - cosi;
                    rdir = -norm(-tr.dir + hy * normal);
                    refracted = true;
                }
            }
            if (refracted) {
                tr.origin = tr.hit;
                tr.dir = rdir;
            } else {
                tr.origin = tr.hit;
                tr.dir = reflection(normal, -tr.dir);
            }
        } else {
            float reflectprob = luma(albedocolor);
            if (!RandomBool(reflectprob)) {
                break;
            }
            Spectrum albedospec = Spectrum::ReflectionSpectrum(albedocolor);
            FDisp outdir;
            GenerateLambertian(normal, outdir);
            float cosi = -dot(tr.dir, normal);
            float coso = dot(outdir, normal);
            float cosio = -dot(tr.dir, outdir);
            float al = 1 - 0.75f * roughness / (roughness + 0.5f);
            float ac = 0.20f * roughness / (roughness + 0.15f);
            Spectrum afspec = albedospec;
            float b = 0.56f * roughness / (roughness + 0.11f);
            float s = cosio - cosi * coso;
            float t = 1;
            if (s > 0) {
                t = std::max(cosi, coso);
            }
            float bf = b * s / t;
            fmadd(afspec, ac, al + bf);
            modulate(current, albedospec, 1 / reflectprob);
            modulate(current, afspec);
            tr.origin = tr.hit;
            tr.dir = outdir;
        }
    }
}
