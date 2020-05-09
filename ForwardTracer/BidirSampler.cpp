#define _CRT_SECURE_NO_WARNINGS

#include "BidirSampler.h"
#include "header.h"
#include <cmath>
#include <cstdio>

static float const PixelPassrate = 1.0f;
constexpr int MaxPathSectionLength = 100;

constexpr float invpi = 0.3183099f;

static bool PathEnabled(int lightnodes, int lensnodes)
{
    return true;
}

static bool PathWeighted(int lightnodes, int lensnodes)
{
    return true;
}

void BidirSampler::PrepareLightStartGenerator()
{
    emitters.resize(scene.FaceCount());
    float totalpower = 0;
    for (int i = 0; i < (int)emitters.size(); ++i) {
        FDisp ab = scene[i].mwt.xunit();
        FDisp ac = scene[i].mwt.yunit();
        FDisp n = cross(ab, ac);
        float l = luma(scene[i].mat.emitgain);
        if (scene[i].mat.emittex < 0) {
            l = 0;
        }
        emitters[i].area = 0.5f * sqrtf(dot(n, n));
        emitters[i].power = l * emitters[i].area;
        emitters[i].dens = l;
        totalpower += emitters[i].power;
    }
    for (int i = 0; i < (int)emitters.size(); ++i) {
        emitters[i].fraction = emitters[i].power / totalpower;
        emitters[i].dens /= totalpower;
        if (emitters[i].dens > 0) {
            emitters[i].contribution = Spectrum::One();
            modulate(emitters[i].contribution, 1 / emitters[i].dens);
        } else {
            emitters[i].contribution = Spectrum::Zero();
        }
    }
}

void BidirSampler::GenerateTriangleInterior(float& u, float& v)
{
    GenerateTriangle(u);
    u = fabsf(u);
    GenerateUniform(v);
    v *= 1 - u;
}

float BidirSampler::FresnelHelper(float refrindex, float cosi, float fresnel_f)
{
    if (fresnel_f > 0) {
        float g = sqrtf(fresnel_f);
        float t1 = (g - cosi) / (g + cosi);
        float t2 = (cosi * (g + cosi) - 1) / (cosi * (g - cosi) + 1);
        return 0.5f * t1 * t1 * (1 + t2 * t2);
    } else {
        return 1;
    }
}

void BidirSampler::ModulateOrenNayarGain(
    Spectrum& spec, Spectrum const& albedo, float cosi, float coso, float cosio, float roughness)
{
    float al = 1 - 0.75f * roughness / (roughness + 0.5f);
    float ac = 0.20f * roughness / (roughness + 0.15f);
    float b = 0.56f * roughness / (roughness + 0.11f);
    float s = cosio - cosi * coso;
    float t = 1;
    if (s > 0) {
        t = fmaxf(cosi, coso);
    }
    float bf = b * s / t;
    Spectrum gain = albedo;
    fmadd(gain, ac, al + bf);
    modulate(spec, gain);
}

void BidirSampler::ExtendPath(std::vector<Event>& path, Spectrum contribution)
{
    TraceRequest tr;
    tr.origin = path[0].origin;
    tr.dir = path[0].outdir;
    while (path.size() < MaxPathSectionLength && Trace(tr)) {
        path.emplace_back();
        Event& e = path.back();
        Event& prev = path[path.size() - 2];
        e.cat = prev.cat;
        e.index = prev.index + 1;
        e.origin = tr.hit;
        e.indir = -tr.dir;
        e.face = tr.face;
        e.tu = dot(e.origin, scene[e.face].texu);
        e.tv = dot(e.origin, scene[e.face].texv);
        e.norm = scene[e.face].mwt.zunit();
        e.contribution = contribution;
        e.geom = Geometric(prev, e, prev.outdir);
        e.pathdensfactor = prev.scatterdens * e.geom;
        e.pathdensdeltaorder = prev.scatterdeltaorder;
        e.mat.emitcolor = scene.SampleColor(scene[e.face].mat.emittex, e.tu, e.tv) * scene[e.face].mat.emitgain;
        e.mat.albedocolor = scene.SampleColor(scene[e.face].mat.albedotex, e.tu, e.tv);
        e.mat.albedoluma = luma(e.mat.albedocolor);
        e.mat.transparency = luma(scene.SampleColor(scene[e.face].mat.transparencytex, e.tu, e.tv));
        e.mat.roughness = luma(scene.SampleColor(scene[e.face].mat.roughnesstex, e.tu, e.tv));
        e.mat.refrindex = scene[e.face].mat.refrindex;
        if (ScatterSample(e, contribution)) {
            tr.origin = e.origin;
            tr.dir = e.outdir;
        } else {
            break;
        }
    }
}

void BidirSampler::GenerateLightSubpath()
{
    int ei = 0;
    while (true) {
        float q;
        GenerateUniform(q);
        while (ei < (int)emitters.size() - 1 && q > emitters[ei].fraction) {
            q -= emitters[ei].fraction;
            ei += 1;
        }
        if (emitters[ei].fraction != 0) {
            break;
        }
    }
    float u;
    float v;
    GenerateTriangleInterior(u, v);
    lightpath.resize(1);
    Event& e = lightpath.front();
    e.cat = EventCat::Light;
    e.index = 0;
    e.face = ei;
    e.origin = scene[ei].mwt * FPoint{ u, v, 0 };
    e.tu = dot(e.origin, scene[ei].texu);
    e.tv = dot(e.origin, scene[ei].texv);
    e.norm = scene[ei].mwt.zunit();
    e.indir = e.norm;
    e.contribution = emitters[ei].contribution;
    e.geom = 1;
    e.pathdensfactor = emitters[ei].dens;
    e.pathdensdeltaorder = 0;
    e.mat.emitcolor = scene.SampleColor(scene[e.face].mat.emittex, e.tu, e.tv) * scene[e.face].mat.emitgain;
    e.mat.albedocolor = FDisp{ 0, 0, 0 };
    e.mat.albedoluma = 0;
    e.mat.transparency = 0;
    e.mat.roughness = 0;
    e.mat.refrindex = 1;
    e.scatterdens = invpi;
    e.scatterdeltaorder = 0;
    GenerateLambertian(e.norm, e.outdir);
    FDisp emitcolor = scene.SampleColor(scene[e.face].mat.emittex, e.tu, e.tv);
    Spectrum contribution = Spectrum::EmissionSpectrum((1 / invpi) * e.mat.emitcolor);
    modulate(contribution, e.contribution);
    ExtendPath(lightpath, contribution);
}

void BidirSampler::GenerateLensSubpath(int ix, int iy)
{
    float dx;
    float dy;
    GenerateTriangle(dx);
    GenerateTriangle(dy);
    float cx = 2.0f * (ix + dx + 0.5f) / width - 1.0f;
    float cy = 1.0f - 2.0f * (iy + dy + 0.5f) / height;
    float cu = cx * camera.utan;
    float cv = cy * camera.vtan;
    lenspath.resize(1);
    Event& e = lenspath.front();
    e.cat = EventCat::Lens;
    e.index = 0;
    e.origin = camera.mwc.origin();
    e.norm = camera.mwc.zunit();
    e.indir = e.norm;
    e.face = -1;
    e.tu = 0;
    e.tv = 0;
    e.contribution = Spectrum::One();
    e.geom = 1;
    e.pathdensfactor = 1;
    e.pathdensdeltaorder = 1;
    e.mat.emitcolor = FDisp{ 0, 0, 0 };
    e.mat.albedocolor = FDisp{ 0, 0, 0 };
    e.mat.albedoluma = 0;
    e.mat.transparency = 0;
    e.mat.roughness = 0;
    e.mat.refrindex = 1;
    e.outdir = norm(camera.mwc * FDisp{ cu, cv, 1.0f });
    {
        float cosa = dot(e.indir, e.outdir);
        float cosasqr = cosa * cosa;
        float cosabisqr = cosasqr * cosasqr;
        e.scatterdens = 1.0f / (4.0f * camera.utan * camera.vtan * cosabisqr);
        e.scatterdeltaorder = 0;
    }
    ExtendPath(lenspath, Spectrum::One());
}

float BidirSampler::Geometric(Event const& from, Event const& to, FDisp dir)
{
    FDisp delta = to.origin - from.origin;
    float cosa = dot(from.norm, dir);
    float cosb = -dot(to.norm, dir);
    float distsqr = dot(delta, delta);
    return cosa * cosb / distsqr;
}

float BidirSampler::Geometric(Event const& from, Event const& to)
{
    FDisp delta = to.origin - from.origin;
    FDisp dir = norm(delta);
    float cosa = dot(from.norm, dir);
    float cosb = -dot(to.norm, dir);
    float distsqr = dot(delta, delta);
    return cosa * cosb / distsqr;
}

bool BidirSampler::ScatterSample(Event& e, Spectrum& contribution)
{
    if (e.index == 0) {
        return false;
    }
    if (RandomBool(e.mat.transparency)) {
        e.scatterdens = e.mat.transparency;
        FDisp normal = e.norm;
        float refrindex = e.mat.refrindex;
        float cosi = dot(e.indir, e.norm);
        if (cosi < 0) {
            normal = -normal;
            cosi = -cosi;
            refrindex = 1 / refrindex;
        }
        bool refracted = false;
        float t = refrindex * refrindex - 1 + cosi * cosi;
        if (t >= 0) {
            float fresnel = FresnelHelper(refrindex, cosi, t);
            if (RandomBool(1 - fresnel)) {
                float hy = sqrtf(t) - cosi;
                e.outdir = -norm(e.indir + hy * normal);
                e.scatterdens *= 1 - fresnel;
                refracted = true;
            } else {
                e.scatterdens *= fresnel;
            }
        }
        if (!refracted) {
            e.outdir = reflection(normal, e.indir);
        }
        e.scatterdeltaorder = 1;
        return true;
    } else {
        if (!RandomBool(e.mat.albedoluma)) {
            return false;
        }
        FDisp normal = e.norm;
        float cosi = dot(normal, e.indir);
        if (cosi < 0) {
            normal = -normal;
            cosi = -cosi;
        }
        GenerateLambertian(normal, e.outdir);
        e.scatterdens = (1 - e.mat.transparency) * e.mat.albedoluma * invpi;
        e.scatterdeltaorder = 0;
        float coso = dot(normal, e.outdir);
        float cosio = dot(e.indir, e.outdir);
        Spectrum albedospec = Spectrum::ReflectionSpectrum(e.mat.albedocolor);
        ModulateOrenNayarGain(contribution, albedospec, cosi, coso, cosio, e.mat.roughness);
        modulate(contribution, albedospec, 1 / e.mat.albedoluma);
        return true;
    }
}

float BidirSampler::ScatterDensity(Event const& e, FDisp indir, FDisp outdir)
{
    if (e.index == 0 && e.cat == EventCat::Light) {
        float cosi = dot(e.norm, indir);
        float coso = dot(e.norm, outdir);
        if (cosi > 0 && coso > 0) {
            return invpi;
        } else {
            return 0;
        }
    } else if (e.index == 0 && e.cat == EventCat::Lens) {
        float cosa = dot(indir, outdir);
        if (cosa <= 0) {
            return 0;
        } else {
            float cosasqr = cosa * cosa;
            float cosabisqr = cosasqr * cosasqr;
            return 1.0f / (4.0f * camera.utan * camera.vtan * cosabisqr);
        }
    } else {
        float cosi = dot(e.norm, indir);
        float coso = dot(e.norm, outdir);
        if (cosi * coso > 0) {
            return (1 - e.mat.transparency) * e.mat.albedoluma * invpi;
        } else {
            return 0;
        }
    }
}

float BidirSampler::EmissionDensity(Event const& e, FDisp indir, FDisp outdir)
{
    float cosi = dot(e.norm, indir);
    float coso = dot(e.norm, outdir);
    if (cosi > 0 && coso > 0) {
        return invpi;
    } else {
        return 0;
    }
}

bool BidirSampler::ModulateScatterFlux(Spectrum& flux, Event const& e, FDisp indir, FDisp outdir)
{
    if (e.index == 0 && e.cat == EventCat::Light) {
        modulate(flux, Spectrum::EmissionSpectrum(e.mat.emitcolor));
        return true;
    } else if (e.index == 0 && e.cat == EventCat::Lens) {
        float cosa = dot(indir, outdir);
        if (cosa > 0) {
            float cosasqr = cosa * cosa;
            float cosabisqr = cosasqr * cosasqr;
            float fs = 1.0f / (4.0f * camera.utan * camera.vtan * cosabisqr);
            modulate(flux, fs);
            return true;
        }
        return false;
    } else {
        float cosi = dot(e.norm, indir);
        float coso = dot(e.norm, outdir);
        if (cosi * coso > 0) {
            FDisp color = e.mat.albedocolor;
            if (dot(color, color) != 0) {
                if (cosi < 0) {
                    cosi = -cosi;
                    coso = -coso;
                }
                float cosio = dot(indir, outdir);
                Spectrum albedospec = Spectrum::ReflectionSpectrum(e.mat.albedocolor);
                ModulateOrenNayarGain(flux, albedospec, cosi, coso, cosio, e.mat.roughness);
                modulate(flux, albedospec, (1 - e.mat.transparency) * invpi);
                return true;
            }
        }
        return false;
    }
}

bool BidirSampler::ModulateEmissionFlux(Spectrum& flux, Event const& e, FDisp indir, FDisp outdir)
{
    FDisp color = e.mat.emitcolor;
    if (dot(color, color) != 0) {
        Spectrum emitspec = Spectrum::EmissionSpectrum(color);
        modulate(flux, emitspec);
        return true;
    }
    return false;
}

FDisp BidirSampler::PathContribution(int lightnodes, int lensnodes)
{
    Spectrum rb;
    if (lightnodes == 0) {
        Event& lense = lenspath[lensnodes - 1];
        if (lense.scatterdeltaorder != 0) {
            return FDisp{ 0, 0, 0 };
        }
        rb = lense.contribution;
        if (!ModulateEmissionFlux(rb, lense, lense.norm, lense.outdir)) {
            return FDisp{ 0, 0, 0 };
        }
    } else {
        Event& lighte = lightpath[lightnodes - 1];
        Event& lense = lenspath[lensnodes - 1];
        if (lighte.scatterdeltaorder + lense.scatterdeltaorder != 0) {
            return FDisp{ 0, 0, 0 };
        }
        FDisp conndir = norm(lense.origin - lighte.origin);
        rb = lighte.contribution;
        modulate(rb, lense.contribution, Geometric(lighte, lense, conndir));
        if (!ModulateScatterFlux(rb, lighte, lighte.indir, conndir)) {
            return FDisp{ 0, 0, 0 };
        }
        if (!ModulateScatterFlux(rb, lense, lense.indir, -conndir)) {
            return FDisp{ 0, 0, 0 };
        }
    }
    return rb.Responce();
}

float BidirSampler::PathWeight(int lightnodes, int lensnodes)
{
    float wden = 0;
    wden += 1;
    if (lightnodes >= 1) {
        float dens3 = 1;
        int deltaorder = 0;
        Event& lighte = lightpath[lightnodes - 1];
        Event& lense = lenspath[lensnodes - 1];
        FDisp conndir = norm(lense.origin - lighte.origin);
        {
            float dp = ScatterDensity(lense, lense.indir, -conndir);
            float geom = Geometric(lighte, lense, conndir);
            dens3 *= dp * geom / lighte.pathdensfactor;
            deltaorder += - lighte.pathdensdeltaorder;
            if (deltaorder == 0 && PathWeighted(lightnodes - 1, lensnodes + 1)) {
                wden += dens3 * dens3;
            }
        }
        if (lightnodes >= 2) {
            float dp = ScatterDensity(lighte, lighte.indir, conndir);
            float geom = lighte.geom;
            dens3 *= dp * geom / lightpath[lightnodes - 2].pathdensfactor;
            deltaorder += - lightpath[lightnodes - 2].pathdensdeltaorder;
            if (deltaorder == 0 && PathWeighted(lightnodes - 2, lensnodes + 2)) {
                wden += dens3 * dens3;
            }
        }
        for (int i = lightnodes - 3; i >= 0; --i) {
            Event& e = lightpath[i + 1];
            float dp = e.scatterdens;
            float geom = e.geom;
            dens3 *= dp * geom / lightpath[i].pathdensfactor;
            deltaorder += e.scatterdeltaorder - lightpath[i].pathdensdeltaorder;
            if (deltaorder == 0 && PathWeighted(i, lensnodes + lightnodes - i)) {
                wden += dens3 * dens3;
            }
        }
    }
    if (lensnodes >= 2) {
        float dens3 = 1;
        int deltaorder = 0;
        Event& lense = lenspath[lensnodes - 1];
        FDisp conndir;
        {
            if (lightnodes >= 1) {
                Event& lighte = lightpath[lightnodes - 1];
                conndir = norm(lense.origin - lighte.origin);
                float dp = ScatterDensity(lighte, lighte.indir, conndir);
                float geom = Geometric(lighte, lense, conndir);
                dens3 *= dp * geom / lense.pathdensfactor;
                deltaorder += - lense.pathdensdeltaorder;
            } else {
                conndir = -lense.norm;
                float dens = emitters[lense.face].dens;
                dens3 *= dens / lense.pathdensfactor;
                deltaorder += - lense.pathdensdeltaorder;
            }
            if (deltaorder == 0 && PathWeighted(lightnodes + 1, lensnodes - 1)) {
                wden += dens3 * dens3;
            }
        }
        if (lensnodes >= 3) {
            float dp;
            if (lightnodes >= 1) {
                dp = ScatterDensity(lense, lense.indir, -conndir);
            } else {
                dp = EmissionDensity(lense, lense.norm, -conndir);
            }
            float geom = lense.geom;
            dens3 *= dp * geom / lenspath[lensnodes - 2].pathdensfactor;
            deltaorder += - lenspath[lensnodes - 2].pathdensdeltaorder;
            if (deltaorder == 0 && PathWeighted(lightnodes + 2, lensnodes - 2)) {
                wden += dens3 * dens3;
            }
        }
        for (int i = lensnodes - 3; i >= 1; --i) {
            Event& e = lenspath[i + 1];
            float dp = e.scatterdens;
            float geom = e.geom;
            dens3 *= dp * geom / lenspath[i].pathdensfactor;
            deltaorder += e.scatterdeltaorder - lenspath[i].pathdensdeltaorder;
            if (deltaorder == 0 && PathWeighted(lensnodes + lightnodes - i, i)) {
                wden += dens3 * dens3;
            }
        }
    }
    if (wden > 0) {
        return 1 / wden;
    } else {
        return 0;
    }
}

FDisp BidirSampler::GetCurrentValue(FDisp dir)
{
    FDisp cdir = camera.mcw * dir;
    float cdx, cdy, cdz;
    cdir.unpack(cdx, cdy, cdz);
    if (cdz >= 0) {
        return FDisp{ 0, 0, 0 };
    }
    float cu = cdx / cdz;
    float cv = cdy / cdz;
    float cx = cu / camera.utan;
    float cy = cv / camera.vtan;
    float nx = 0.5f * (1.0f + cx) * width;
    float ny = 0.5f * (1.0f - cy) * height;
    int ix = (int)floorf(nx);
    int iy = (int)floorf(ny);
    return GetValue(ix, iy);
}

void BidirSampler::RecordContribution(FDisp dir, FDisp value)
{
    FDisp cdir = camera.mcw * dir;
    float cdx, cdy, cdz;
    cdir.unpack(cdx, cdy, cdz);
    if (cdz >= 0) {
        return;
    }
    float cu = cdx / cdz;
    float cv = cdy / cdz;
    float cx = cu / camera.utan;
    float cy = cv / camera.vtan;
    float nx = 0.5f * (1.0f + cx) * width - 0.5f;
    float ny = 0.5f * (1.0f - cy) * height - 0.5f;
    float nfx = floorf(nx);
    float nfy = floorf(ny);
    float dx = nx - nfx;
    float dy = ny - nfy;
    int ix = (int)nfx;
    int iy = (int)nfy;
    value *= 1 / PixelPassrate;
    RecordToFrame(ix, iy, value * (1 - dx) * (1 - dy));
    RecordToFrame(ix + 1, iy, value * dx * (1 - dy));
    RecordToFrame(ix, iy + 1, value * (1 - dx) * dy);
    RecordToFrame(ix + 1, iy + 1, value * dx * dy);
    /*float q;
    GenerateUniform(q);
    if (q < dx) {
        ix += 1;
    }
    GenerateUniform(q);
    if (q < dy) {
        iy += 1;
    }
    RecordToFrame(ix, iy, value);*/
}

void BidirSampler::RecordPathContribution(int lightnodes, int lensnodes)
{
    if (lightnodes < 0 || lensnodes <= 0 || lightnodes + lensnodes < 2) {
        return;
    }
    if (!PathEnabled(lightnodes, lensnodes)) {
        return;
    }
    FDisp contribution = PathContribution(lightnodes, lensnodes);
    float cluma = luma(contribution);
    if (cluma == 0) {
        return;
    }
    float weight = PathWeight(lightnodes, lensnodes);
    if (isnan(cluma)) {
        DumpPath(lightnodes, lensnodes);
        throw "shit";
    }
    if (isnan(weight)) {
        DumpPath(lightnodes, lensnodes);
        throw "damnit";
    }
    contribution *= weight;
    FDisp lensdir;
    if (lensnodes >= 2) {
        lensdir = lenspath[0].origin - lenspath[1].origin;
    } else {
        lensdir = lenspath[0].origin - lightpath[lightnodes - 1].origin;
    }
    if (lightnodes != 0 && lensnodes != 0) {
        if (!TestVisible(lightpath[lightnodes - 1].origin, lenspath[lensnodes - 1].origin)) {
            return;
        }
    }
    RecordContribution(lensdir, contribution);
}

void BidirSampler::DumpPath(int lightnodes, int lensnodes)
{
    FILE* fout = fopen("D:\\rt\\error.txt", "w");
    auto DumpVector = [&](char const* name, FXmmQuad const& q)
    {
        float x, y, z, w;
        q.unpack(x, y, z, w);
        fprintf(fout, "%s: {%12.6g, %12.6g, %12.6g, %12.6g}\n", name, x, y, z, w);
    };
    auto DumpEvent = [&](Event& e)
    {
        if (e.cat == EventCat::Light) {
            fprintf(fout, "        cat: Light\n");
        } else if (e.cat == EventCat::Lens) {
            fprintf(fout, "        cat: Lens\n");
        }
        fprintf(fout, "        index: %i\n", e.index);
        DumpVector(   "        origin", e.origin);
        DumpVector(   "        norm", e.norm);
        DumpVector(   "        indir", e.indir);
        DumpVector(   "        outdir", e.outdir);
        DumpVector(   "        contribution.Responce()", e.contribution.Responce());
        fprintf(fout, "        face: %i\n", e.face);
        fprintf(fout, "        tu: %12.6g\n", e.tu);
        fprintf(fout, "        tv: %12.6g\n", e.tv);
        DumpVector(   "        mat.emitcolor", e.mat.emitcolor);
        DumpVector(   "        mat.albedocolor", e.mat.albedocolor);
        fprintf(fout, "        mat.transparency: %12.6g\n", e.mat.transparency);
        fprintf(fout, "        mat.roughness: %12.6g\n", e.mat.roughness);
        fprintf(fout, "        mat.refrindex: %12.6g\n", e.mat.refrindex);
        fprintf(fout, "        mat.albedoluma: %12.6g\n", e.mat.albedoluma);
        fprintf(fout, "        geom: %12.6g\n", e.geom);
        fprintf(fout, "        scatterdens: %12.6g\n", e.scatterdens);
        fprintf(fout, "        scatterdeltaorder: %i\n", e.scatterdeltaorder);
        fprintf(fout, "        pathdensfactor: %12.6g\n", e.pathdensfactor);
        fprintf(fout, "        pathdensdeltaorder: %i\n", e.pathdensdeltaorder);
    };
    fprintf(fout, "lightpath:\n");
    for (int i = 0; i < lightnodes; ++i) {
        fprintf(fout, "    %i:\n", i);
        DumpEvent(lightpath[i]);
    }
    fprintf(fout, "lenspath:\n");
    for (int i = 0; i < lensnodes; ++i) {
        fprintf(fout, "    %i:\n", i);
        DumpEvent(lenspath[i]);
    }
    fclose(fout);
}

BidirSampler::BidirSampler(int width, int height)
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
    PrepareLightStartGenerator();
}

BidirSampler::~BidirSampler()
{
}

void BidirSampler::IteratePixel(int ix, int iy)
{
    GenerateLightSubpath();
    if (RandomBool(PixelPassrate)) {
        GenerateLensSubpath(ix, iy);
        for (int li = 0; li  <= (int)lightpath.size(); ++li) {
            for (int ei = 1; ei <= (int)lenspath.size(); ++ei) {
                RecordPathContribution(li, ei);
            }
        }
    }
}
