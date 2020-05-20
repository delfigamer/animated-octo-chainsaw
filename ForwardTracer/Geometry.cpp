#define _CRT_SECURE_NO_WARNINGS

#include "Geometry.h"
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <array>
#include <cstdio>
#include <cmath>

inline void ToLower(char* buf)
{
    for (unsigned char* ubuf = (unsigned char*)buf; *ubuf != 0; ++ubuf) {
        int ch = *ubuf;
        if (ch >= 'A' && ch <= 'Z') {
            *ubuf = ch + ('a' - 'A');
        }
    }
}

inline void ToLower(std::string& str)
{
    for (char& ref : str) {
        int ch = (int)ref;
        if (ch >= 'A' && ch <= 'Z') {
            ref = (char)(ch + ('a' - 'A'));
        }
    }
}

struct LoaderMaterial
{
    FDisp emitgain = { 1, 1, 1 };
    float refrindex = 1;
    std::string emittex;
    std::string albedotex;
    std::string transparencytex;
    std::string roughnesstex;
    int transmittancespec = -1;
    float transmittancelength = 1;
};

struct MatLoader
{
    Geometry& target;
    std::map<std::string, LoaderMaterial> materials;
    LoaderMaterial* current = nullptr;
    std::map<std::string, int> textures;

    MatLoader(Geometry& target)
        : target(target)
    {
    }

    void TryBeginMaterial(std::string const& line)
    {
        char namebuf[256] = "";
        if (sscanf(line.c_str(), " >> %256s", namebuf) != 0) {
            ToLower(namebuf);
            current = &materials[namebuf];
        }
    }

    void TryTexture(std::string const& line)
    {
        if (!current) {
            return;
        }
        char pathbuf[256] = "";
        if (sscanf(line.c_str(), " Emit: %256s", pathbuf) != 0) {
            ToLower(pathbuf);
            current->emittex = pathbuf;
        }
        if (sscanf(line.c_str(), " Albedo: %256s", pathbuf) != 0) {
            ToLower(pathbuf);
            current->albedotex = pathbuf;
        }
        if (sscanf(line.c_str(), " Transparency: %256s", pathbuf) != 0) {
            ToLower(pathbuf);
            current->transparencytex = pathbuf;
        }
        if (sscanf(line.c_str(), " Roughness: %256s", pathbuf) != 0) {
            ToLower(pathbuf);
            current->roughnesstex = pathbuf;
        }
    }

    void TryParam(std::string const& line)
    {
        if (!current) {
            return;
        }
        float x, y, z;
        int i;
        if (sscanf(line.c_str(), " EmitGain:%f,%f,%f", &x, &y, &z) != 0) {
            current->emitgain = FDisp{ x, y, z };
        }
        if (sscanf(line.c_str(), " RIndex:%f", &x) != 0) {
            current->refrindex = x;
        }
        if (sscanf(line.c_str(), " TransmittanceSpectrum:%i", &i) != 0) {
            current->transmittancespec = i - 1;
        }
        if (sscanf(line.c_str(), " TransmittanceLength:%f", &x) != 0) {
            current->transmittancelength = x;
        }
    }

    void ProcessLine(std::string const& line)
    {
        TryBeginMaterial(line);
        TryTexture(line);
        TryParam(line);
    }

    void Process(char const* path)
    {
        std::ifstream input(path);
        int row = 0;
        while (!input.eof()) {
            std::string line;
            std::getline(input, line);
            row += 1;
            if (!line.empty()) {
                ProcessLine(line);
            }
        }
    }

    LoaderMaterial* operator[](std::string name)
    {
        ToLower(name);
        auto it = materials.find(name);
        if (it == materials.end()) {
            return nullptr;
        } else {
            return &it->second;
        }
    }

    int LoadTexture(std::string const& path)
    {
        if (path.empty()) {
            return -1;
        }
        auto it = textures.find(path);
        if (it == textures.end()) {
            ColorTexture tex;
            tex.Load(path.c_str());
            if (tex.Empty()) {
                textures[path] = -1;
                return -1;
            } else {
                target.textures.push_back(std::move(tex));
                int id = (int)target.textures.size() - 1;
                textures[path] = id;
                return id;
            }
        } else {
            return it->second;
        }
    }
};

struct T3DLoader
{
    Geometry& target;
    MatLoader matloader;
    int trindex = -1;
    LoaderMaterial* currentmat = nullptr;
    FPoint points[3];
    int ptindex;
    FDisp texu;
    FDisp texv;
    FPoint texorigin;
    float panu;
    float panv;

    T3DLoader(Geometry& target)
        : target(target)
        , matloader(target)
    {
    }

    void TryBeginPolygon(std::string const& line)
    {
        unsigned int flags = 0;
        char texturebuf[256] = "";
        if (false
            || sscanf(line.c_str(), " Begin Polygon Item=%*s Texture=%256s Flags=%u", texturebuf, &flags) != 0
            || sscanf(line.c_str(), " Begin Polygon Texture=%256s Flags=%u", texturebuf, &flags) != 0
            || sscanf(line.c_str(), " Begin Polygon Item=%*s Flags=%u", &flags) != 0
            || sscanf(line.c_str(), " Begin Polygon Flags=%u", &flags) != 0
        ) {
            trindex += 1;
            ptindex = 0;
            texu = FDisp{ 0, 0, 0 };
            texv = FDisp{ 0, 0, 0 };
            panu = 0;
            panv = 0;
            currentmat = matloader[texturebuf];
        }
    }

    void TryTextureCoord(std::string const& line)
    {
        float x, y, z;
        if (trindex >= 0 && sscanf(line.c_str(), " Origin%f ,%f, %f", &x, &y, &z) != 0) {
            texorigin = FPoint{ -x, y, z };
        }
        if (trindex >= 0 && sscanf(line.c_str(), " TextureU%f ,%f, %f", &x, &y, &z) != 0) {
            texu = FDisp{ -x, y, z };
        }
        if (trindex >= 0 && sscanf(line.c_str(), " TextureV%f ,%f, %f", &x, &y, &z) != 0) {
            texv = FDisp{ -x, y, z };
        }
        if (trindex >= 0 && sscanf(line.c_str(), " Pan U=%f V=%f", &x, &y) != 0) {
            panu = x;
            panv = y;
        }
    }

    void TryVertex(std::string const& line)
    {
        float x, y, z;
        if (trindex >= 0 && sscanf(line.c_str(), " Vertex%f ,%f, %f", &x, &y, &z) != 0) {
            if (ptindex < 2) {
                points[ptindex] = FPoint{ -x, y, z };
                ptindex += 1;
            } else {
                points[2] = FPoint{ -x, y, z };
                target.faces.emplace_back();
                if (!BuildFace(target.faces.back())) {
                    target.faces.pop_back();
                }
                points[1] = points[2];
            }
        }
    }

    FXmmQuad Covector(FDisp norm, float offset)
    {
        float x, y, z;
        norm.unpack(x, y, z);
        return FXmmQuad{ x, y, z, offset };
    }

    bool BuildFace(Face& face)
    {
        if (!currentmat) {
            return false;
        }
        FDisp ab = points[1] - points[0];
        FDisp ac = points[2] - points[0];
        FDisp an = norm(cross(ab, ac));
        FMat mwt = FMat::basis(ab, ac, an, points[0]);
        FMat mtw = inverse(mwt);
        FMat mww = mwt * mtw;
        mww = mww - FMat::identity();
        float err = sqr(mww);
        face = Face{ mwt, mtw };
        if (err > 1.0e-3) {
            return false;
        }
        face.mat.emitgain = currentmat->emitgain;
        face.mat.refrindex = currentmat->refrindex;
        face.mat.emittex = matloader.LoadTexture(currentmat->emittex);
        face.mat.albedotex = matloader.LoadTexture(currentmat->albedotex);
        face.mat.transparencytex = matloader.LoadTexture(currentmat->transparencytex);
        face.mat.roughnesstex = matloader.LoadTexture(currentmat->roughnesstex);
        face.mat.transmittancespec = currentmat->transmittancespec;
        if (face.mat.transmittancespec >= Spectrum::TableSize()) {
            face.mat.transmittancespec = -1;
        }
        face.mat.transmittancelength = currentmat->transmittancelength;
        face.texu = Covector(texu, panu - dot(texorigin, texu));
        face.texv = Covector(texv, panv - dot(texorigin, texv));
        return true;
    }

    void ProcessLine(std::string const& line)
    {
        TryBeginPolygon(line);
        TryTextureCoord(line);
        TryVertex(line);
    }

    void Process(char const* path)
    {
        std::ifstream input(path);
        int row = 0;
        while (!input.eof()) {
            std::string line;
            std::getline(input, line);
            row += 1;
            if (!line.empty()) {
                ProcessLine(line);
            }
        }
    }
};

Geometry::Geometry()
{
}

Geometry::~Geometry()
{
}

void Geometry::LoadFromT3D(char const* path)
{
    T3DLoader loader{ *this };
    loader.matloader.Process("D:\\rt\\materials.txt");
    loader.Process(path);
}

bool Geometry::Trace(TraceRequest& tr)
{
    tr.param = INFINITY;
    tr.face = -1;
    for (int i = 0; i < faces.size(); ++i) {
        Face& face = faces[i];
        FPoint torigin = face.mtw * tr.origin;
        float tox, toy, toz;
        torigin.unpack(tox, toy, toz);
        FDisp tdir = face.mtw * tr.dir;
        float tdx, tdy, tdz;
        tdir.unpack(tdx, tdy, tdz);
        float tparam = -toz / tdz;
        if (tparam <= 0.001f || tparam >= tr.param) {
            continue;
        }
        FPoint thit = torigin + tparam * tdir;
        float thx, thy, thz;
        thit.unpack(thx, thy, thz);
        if (thx < 0 || thy < 0 || (thx + thy) > 1) {
            continue;
        }
        tr.param = tparam;
        tr.hit = tr.origin + tr.dir * tparam;
        tr.hitlocal = thit;
        tr.face = i;
        if (toz >= 0) {
            tr.side = 1;
        } else {
            tr.side = -1;
        }
    }
    return isfinite(tr.param);
}

bool Geometry::Test(FPoint const& origin, FDisp const& delta)
{
    for (int i = 0; i < faces.size(); ++i) {
        Face& face = faces[i];
        FPoint torigin = face.mtw * origin;
        float tox, toy, toz;
        torigin.unpack(tox, toy, toz);
        FDisp tdir = face.mtw * delta;
        float tdx, tdy, tdz;
        tdir.unpack(tdx, tdy, tdz);
        float tparam = -toz / tdz;
        if (tparam <= 0.001f || tparam >= 0.999f) {
            continue;
        }
        FPoint thit = torigin + tparam * tdir;
        float thx, thy, thz;
        thit.unpack(thx, thy, thz);
        if (thx < 0 || thy < 0 || (thx + thy) > 1) {
            continue;
        }
        return false;
    }
    return true;
}

Face& Geometry::operator[](int index)
{
    return faces[index];
}

int Geometry::FaceCount()
{
    return (int)faces.size();
}

FDisp Geometry::SampleColor(int texture, float u, float v)
{
    if (texture < 0) {
        return FDisp{ 0, 0, 0 };
    } else {
        return textures[texture].Sample(u, v);
    }
}
