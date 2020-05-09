#pragma once

#include "SamplerBase.h"
#include "Camera.h"
#include "FLin.h"

class BidirSampler: public SamplerBase
{
public:
    enum class EventCat
    {
        Light,
        Lens,
    };

private:
    struct LocalMaterial
    {
        FDisp emitcolor;
        FDisp albedocolor;
        float transparency;
        float roughness;
        float refrindex;
        float albedoluma;
    };

    struct Event
    {
        EventCat cat;
        int index;
        FPoint origin;
        FDisp norm;
        FDisp indir;
        FDisp outdir;
        Spectrum contribution;
        int face;
        float tu;
        float tv;
        LocalMaterial mat;
        float geom;
        float scatterdens;
        int scatterdeltaorder;
        float pathdensfactor;
        int pathdensdeltaorder;
    };

    struct Emitter
    {
        Spectrum contribution;
        float fraction;
        float area;
        float power;
        float dens;
    };

private:
    Camera camera;
    std::vector<Emitter> emitters;
    std::vector<Event> lightpath;
    std::vector<Event> lenspath;

    void PrepareLightStartGenerator();
    void GenerateTriangleInterior(float& u, float& v);
    float FresnelHelper(float refrindex, float cosi, float fresnel_f);
    void ModulateOrenNayarGain(Spectrum& spec, Spectrum const& albedo, float cosi, float coso, float cosio, float roughness);
    void ExtendPath(std::vector<Event>& path, Spectrum contribution);
    void GenerateLightSubpath();
    void GenerateLensSubpath(int ix, int iy);
    float Geometric(Event const& from, Event const& to, FDisp dir);
    float Geometric(Event const& from, Event const& to);
    bool ScatterSample(Event& e, Spectrum& contribution);
    float ScatterDensity(Event const& e, FDisp indir, FDisp outdir);
    float EmissionDensity(Event const& e, FDisp indir, FDisp outdir);
    bool ModulateScatterFlux(Spectrum& flux, Event const& e, FDisp indir, FDisp outdir);
    bool ModulateEmissionFlux(Spectrum& flux, Event const& e, FDisp indir, FDisp outdir);
    FDisp PathContribution(int lightnodes, int lensnodes);
    float PathWeight(int lightnodes, int lensnodes);
    FDisp GetCurrentValue(FDisp dir);
    void RecordContribution(FDisp dir, FDisp value);
    void RecordPathContribution(int lightnodes, int lensnodes);
    void DumpPath(int lightnodes, int lensnodes);

public:
    BidirSampler(int width, int height);
    ~BidirSampler();

    virtual void IteratePixel(int ix, int iy) override;
};
