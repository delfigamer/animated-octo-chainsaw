#define _CRT_SECURE_NO_WARNINGS

#include "Geometry.h"
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <array>
#include <cstdio>
#include <cmath>
#include <random>

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
        if (sscanf(line.c_str(), " >> %255s", namebuf) != 0) {
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
        if (sscanf(line.c_str(), " Emit: %255s", pathbuf) != 0) {
            ToLower(pathbuf);
            current->emittex = pathbuf;
        }
        if (sscanf(line.c_str(), " Albedo: %255s", pathbuf) != 0) {
            ToLower(pathbuf);
            current->albedotex = pathbuf;
        }
        if (sscanf(line.c_str(), " Transparency: %255s", pathbuf) != 0) {
            ToLower(pathbuf);
            current->transparencytex = pathbuf;
        }
        if (sscanf(line.c_str(), " Roughness: %255s", pathbuf) != 0) {
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
    std::set<std::array<int, 6>> segments;

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
            || sscanf(line.c_str(), " Begin Polygon Item=%*s Texture=%255s Flags=%u", texturebuf, &flags) != 0
            || sscanf(line.c_str(), " Begin Polygon Texture=%255s Flags=%u", texturebuf, &flags) != 0
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
                } else {
                    SetSegment(points[0], points[1]);
                    SetSegment(points[0], points[2]);
                    SetSegment(points[1], points[2]);
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

    void SetSegment(FPoint a, FPoint b)
    {
        float ax, ay, az, bx, by, bz;
        a.unpack(ax, ay, az);
        b.unpack(bx, by, bz);
        std::array<int, 6> c = { (int)ax, (int)ay, (int)az, (int)bx, (int)by, (int)bz };
        if (segments.count(c) > 0) {
            return;
        }
        target.edges.push_back(Segment{
            FPoint{ (float)c[0], (float)c[1], (float)c[2] },
            FDisp{ (float)(c[3] - c[0]), (float)(c[4] - c[1]), (float)(c[5] - c[2]) } });
        segments.insert(c);
        std::array<int, 6> cr = { c[3], c[4], c[5], c[0], c[1], c[2] };
        segments.insert(cr);
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
    faces.shrink_to_fit();
    edges.shrink_to_fit();
    textures.shrink_to_fit();
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

// https://www.geometrictools.com/Samples/Distance.html#DistanceSegments3
static float GetClampedRoot(float slope, float h0, float h1)
{
    float r;
    if (h0 < 0) {
        if (h1 > 0) {
            r = -h0 / slope;
            if (r > 1) {
                r = 0.5;
            }
        } else {
            r = 1;
        }
    } else {
        r = 0;
    }
    return r;
}

static void ComputeIntersection(
    float b, float f00, float f10, float sValue[2], int classify[2],
    int edge[2], float end[4])
{
    if (classify[0] < 0) {
        edge[0] = 0;
        end[0] = 0;
        end[1] = f00 / b;
        if (end[1] < 0 || end[1] > 1) {
            end[1] = 0.5;
        }
        if (classify[1] == 0) {
            edge[1] = 3;
            end[2] = sValue[1];
            end[3] = 1;
        } else {
            edge[1] = 1;
            end[2] = 1;
            end[3] = f10 / b;
            if (end[3] < 0 || end[3] > 1) {
                end[3] = 0.5;
            }
        }
    } else if (classify[0] == 0) {
        edge[0] = 2;
        end[0] = sValue[0];
        end[1] = 0;

        if (classify[1] < 0) {
            edge[1] = 0;
            end[2] = 0;
            end[3] = f00 / b;
            if (end[3] < 0 || end[3] > 1) {
                end[3] = 0.5;
            }
        } else if (classify[1] == 0) {
            edge[1] = 3;
            end[2] = sValue[1];
            end[3] = 1;
        } else {
            edge[1] = 1;
            end[2] = 1;
            end[3] = f10 / b;
            if (end[3] < 0 || end[3] > 1) {
                end[3] = 0.5;
            }
        }
    } else {
        edge[0] = 1;
        end[0] = 1;
        end[1] = f10 / b;
        if (end[1] < 0 || end[1] > 1) {
            end[1] = 0.5;
        }

        if (classify[1] == 0) {
            edge[1] = 3;
            end[2] = sValue[1];
            end[3] = 1;
        } else {
            edge[1] = 0;
            end[2] = 0;
            end[3] = f00 / b;
            if (end[3] < 0 || end[3] > 1) {
                end[3] = 0.5;
            }
        }
    }
}

void ComputeMinimumParameters(
    float b, float c, float e, float g00, float g10, float g01, float g11,
    int edge[2], float end[4], float parameter[3])
{
    float delta = end[3] - end[1];
    float h0 = delta * (-b * end[0] + c * end[1] - e);
    if (h0 >= 0) {
        if (edge[0] == 0) {
            parameter[0] = 0;
            parameter[1] = GetClampedRoot(c, g00, g01);
        } else if (edge[0] == 1) {
            parameter[0] = 1;
            parameter[1] = GetClampedRoot(c, g10, g11);
        } else {
            parameter[0] = end[0];
            parameter[1] = end[1];
        }
    } else {
        float h1 = delta * (-b * end[2] + c * end[3] - e);
        if (h1 <= 0) {
            if (edge[1] == 0) {
                parameter[0] = 0;
                parameter[1] = GetClampedRoot(c, g00, g01);
            } else if (edge[1] == 1) {
                parameter[0] = 1;
                parameter[1] = GetClampedRoot(c, g10, g11);
            } else {
                parameter[0] = end[2];
                parameter[1] = end[3];
            }
        } else {
            float z = fminf(fmaxf(h0 / (h0 - h1), 0), 1);
            float omz = 1 - z;
            parameter[0] = omz * end[0] + z * end[2];
            parameter[1] = omz * end[1] + z * end[3];
        }
    }
}

static void DistanceQuery(FPoint ao, FDisp ad, FPoint bo, FDisp bd, float& aw, float& bw)
{
    FDisp P1mP0 = ad;
    FDisp Q1mQ0 = bd;
    FDisp P0mQ0 = ao - bo;
    float a = dot(P1mP0, P1mP0);
    float b = dot(P1mP0, Q1mQ0);
    float c = dot(Q1mQ0, Q1mQ0);
    float d = dot(P1mP0, P0mQ0);
    float e = dot(Q1mQ0, P0mQ0);
    float f00 = d;
    float f10 = f00 + a;
    float f01 = f00 - b;
    float f11 = f10 - b;
    float g00 = -e;
    float g01 = g00 + c;
    float g10 = g00 - b;
    float g11 = g10 + c;
    float parameter[2];
    if (a > 0 && c > 0) {
        float sValue[2];
        sValue[0] = GetClampedRoot(a, f00, f10);
        sValue[1] = GetClampedRoot(a, f01, f11);
        int classify[2];
        for (int i = 0; i < 2; ++i) {
            if (sValue[i] <= 0) {
                classify[i] = -1;
            } else if (sValue[i] >= 1) {
                classify[i] = 1;
            } else {
                classify[i] = 0;
            }
        }
        if (classify[0] == -1 && classify[1] == -1) {
            parameter[0] = 0;
            parameter[1] = GetClampedRoot(c, g00, g01);
        } else if (classify[0] == 1 && classify[1] == 1) {
            parameter[0] = 1;
            parameter[1] = GetClampedRoot(c, g10, g11);
        } else {
            int edge[2];
            float end[4];
            ComputeIntersection(b, f00, f10, sValue, classify, edge, end);
            ComputeMinimumParameters(b, c, e, g00, g10, g01, g11, edge, end, parameter);
        }
    } else {
        if (a > 0) {
            parameter[0] = GetClampedRoot(a, f00, f10);
            parameter[1] = 0;
        } else if (c > 0) {
            parameter[0] = 0;
            parameter[1] = GetClampedRoot(c, g00, g01);
        } else {
            parameter[0] = 0;
            parameter[1] = 0;
        }
    }
    aw = parameter[0];
    bw = parameter[1];
}

FDisp NumericalGradient(FPoint segorigin, FDisp segdelta, FPoint origin, FDisp delta)
{
    FDisp altgrad = FDisp{ 0, 0, 0 };
    float h = 0.1f;
    {
        FDisp delta2 = delta + FDisp{ h, 0, 0 };
        float aw2, bw2;
        DistanceQuery(segorigin, segdelta, origin, delta2, aw2, bw2);
        FPoint apt2 = segorigin + aw2 * segdelta;
        FPoint bpt2 = origin + bw2 * delta2;
        FDisp ab2 = bpt2 - apt2;
        float absqr2 = dot(ab2, ab2);
        altgrad += FDisp{ absqr2 / h, 0, 0 };
    }
    {
        FDisp delta2 = delta + FDisp{ 0, h, 0 };
        float aw2, bw2;
        DistanceQuery(segorigin, segdelta, origin, delta2, aw2, bw2);
        FPoint apt2 = segorigin + aw2 * segdelta;
        FPoint bpt2 = origin + bw2 * delta2;
        FDisp ab2 = bpt2 - apt2;
        float absqr2 = dot(ab2, ab2);
        altgrad += FDisp{ 0, absqr2 / h, 0 };
    }
    {
        FDisp delta2 = delta + FDisp{ 0, 0, h };
        float aw2, bw2;
        DistanceQuery(segorigin, segdelta, origin, delta2, aw2, bw2);
        FPoint apt2 = segorigin + aw2 * segdelta;
        FPoint bpt2 = origin + bw2 * delta2;
        FDisp ab2 = bpt2 - apt2;
        float absqr2 = dot(ab2, ab2);
        altgrad += FDisp{ 0, 0, absqr2 / h };
    }
    {
        FDisp delta2 = delta + FDisp{ -h, 0, 0 };
        float aw2, bw2;
        DistanceQuery(segorigin, segdelta, origin, delta2, aw2, bw2);
        FPoint apt2 = segorigin + aw2 * segdelta;
        FPoint bpt2 = origin + bw2 * delta2;
        FDisp ab2 = bpt2 - apt2;
        float absqr2 = dot(ab2, ab2);
        altgrad -= FDisp{ absqr2 / h, 0, 0 };
    }
    {
        FDisp delta2 = delta + FDisp{ 0, -h, 0 };
        float aw2, bw2;
        DistanceQuery(segorigin, segdelta, origin, delta2, aw2, bw2);
        FPoint apt2 = segorigin + aw2 * segdelta;
        FPoint bpt2 = origin + bw2 * delta2;
        FDisp ab2 = bpt2 - apt2;
        float absqr2 = dot(ab2, ab2);
        altgrad -= FDisp{ 0, absqr2 / h, 0 };
    }
    {
        FDisp delta2 = delta + FDisp{ 0, 0, -h };
        float aw2, bw2;
        DistanceQuery(segorigin, segdelta, origin, delta2, aw2, bw2);
        FPoint apt2 = segorigin + aw2 * segdelta;
        FPoint bpt2 = origin + bw2 * delta2;
        FDisp ab2 = bpt2 - apt2;
        float absqr2 = dot(ab2, ab2);
        altgrad -= FDisp{ 0, 0, absqr2 / h };
    }
    return 0.5f * altgrad;
}

void Geometry::EdgeDistance(FPoint const& origin, FDisp const& delta, float& dist, FDisp& grad)
{
    dist = INFINITY;
    grad = FDisp{ 0, 0, 0 };
    for (Segment const& seg : edges) {
        float aw, bw;
        DistanceQuery(seg.origin, seg.delta, origin, delta, aw, bw);
        FPoint apt = seg.origin + aw * seg.delta;
        FPoint bpt = origin + bw * delta;
        FDisp ab = bpt - apt;
        float absqr = dot(ab, ab);
        if (absqr < dist) {
            dist = absqr;
            grad = FDisp{ 0, 0, 0 };
            FDisp dd = seg.origin - origin;
            if (bw == 0) {
            } else if (bw == 1) {
                if (aw == 0) {
                    grad = -2 * (dd - delta);
                } else if (aw == 1) {
                    grad = -2 * (dd + seg.delta - delta);
                } else {
                    float det = dot(seg.delta, seg.delta);
                    float sr = 2 / det;
                    if (isnormal(sr)) {
                        grad = sr * cross(cross(dd - delta, seg.delta), seg.delta);
                    }
                }
            } else {
                if (aw == 0) {
                    float det = dot(delta, delta);
                    float sr = 2 * dot(dd, delta) / (det * det);
                    if (isnormal(sr)) {
                        grad = sr * cross(cross(dd, delta), delta);
                    }
                } else if (aw == 1) {
                    float det = dot(delta, delta);
                    float sr = 2 * dot(dd + seg.delta, delta) / (det * det);
                    if (isnormal(sr)) {
                        grad = sr * cross(cross(dd + seg.delta, delta), delta);
                    }
                } else {
                    FDisp ar = cross(seg.delta, delta);
                    float alt = dot(dd, ar) * dot(dd, ar) / dot(ar, ar);
                    float arsq = dot(ar, ar);
                    float sr = -2 * dot(cross(seg.delta, dd), ar) * dot(dd, ar) / (arsq * arsq);
                    if (isnormal(sr)) {
                        grad = sr * ar;
                    }
                }
            }
        }
    }
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
