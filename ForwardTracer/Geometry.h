#pragma once

#include "FLin.h"
#include "Texture.h"
#include "Spectrum.h"
#include <vector>

struct FaceMaterial
{
    FDisp emitgain;
    float refrindex;
    int emittex;
    int albedotex;
    int transparencytex;
    int roughnesstex;
    int transmittancespec;
    float transmittancelength;
};

struct Face
{
    FMat mwt;
    FMat mtw;
    FXmmQuad texu;
    FXmmQuad texv;
    FaceMaterial mat;
};

struct TraceRequest
{
// input
    FPoint origin;
    FDisp dir;
// output
    float param;
    FPoint hit;
    int face;
    int side;
};

class Geometry
{
private:
    std::vector<Face> faces;
    std::vector<ColorTexture> textures;

    friend struct MatLoader;
    friend struct T3DLoader;

public:
    Geometry();
    ~Geometry();

    void LoadFromT3D(char const* path);
    bool Trace(TraceRequest& tr);
    bool Test(FPoint const& origin, FDisp const& delta);

    Face& operator[](int index);
    int FaceCount();
    FDisp SampleColor(int texture, float u, float v);
};
