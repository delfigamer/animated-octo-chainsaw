#define _CRT_SECURE_NO_WARNINGS

#include "Spectrum.h"
#include <vector>
#include <memory>
#include <cstdio>

namespace
{
namespace Table
{
    using SPtr = std::unique_ptr<Spectrum>;
    SPtr Zero;
    SPtr One;

    SPtr ResponseR;
    SPtr ResponseG;
    SPtr ResponseB;
    SPtr Luma;

    constexpr int Depth = 24;
    std::vector<SPtr> Emission;
    std::vector<SPtr> Reflection;
    std::vector<SPtr> Other;

    SPtr MakeConstantSpectrum(float x);
    SPtr ReadSpectrum(FILE* f);
    void Initialize();
    int RefIndex(int ri, int gi);
    void FindInterpolation(
        float r, float g, float b,
        int& i1, float& w1,
        int& i2, float& w2,
        int& i3, float& w3);
    Spectrum Interpolate(
        std::vector<SPtr> const& ref,
        float r, float g, float b,
        float scale);
}
}

Table::SPtr Table::MakeConstantSpectrum(float x)
{
    SPtr p(new Spectrum);
    __m128 mx = _mm_set1_ps(x);
    for (int i = 0; i < Spectrum::Size; ++i) {
        p->m[i] = mx;
    }
    return p;
}

Table::SPtr Table::ReadSpectrum(FILE* f)
{
    SPtr p(new Spectrum);
    for (int i = 0; i < Spectrum::Size; ++i) {
        alignas(16) float b[4];
        fscanf(f, "%f%f%f%f", &b[0], &b[1], &b[2], &b[3]);
        p->m[i] = _mm_load_ps(b);
    }
    return p;
}

void Table::Initialize()
{
    static bool Initialized = false;
    if (Initialized) {
        return;
    }
    Zero = MakeConstantSpectrum(0);
    One = MakeConstantSpectrum(1);
#if 0
    FILE* f = fopen("D:\\rt\\rgb.tsv", "r");
    ResponseR = ReadSpectrum(f);
    ResponseG = ReadSpectrum(f);
    ResponseB = ReadSpectrum(f);
    Luma = ReadSpectrum(f);
    fclose(f);
    int count = (Depth + 1) * (Depth + 2) / 2;
    Emission.resize(count);
    f = fopen("D:\\rt\\emit.tsv", "r");
    for (int i = 0; i < count; ++i) {
        Emission[i] = ReadSpectrum(f);
    }
    fclose(f);
    Reflection.resize(count);
    f = fopen("D:\\rt\\reflect.tsv", "r");
    for (int i = 0; i < count; ++i) {
        Reflection[i] = ReadSpectrum(f);
    }
    fclose(f);
    f = fopen("D:\\rt\\extra.tsv", "r");
    while (true) {
        fscanf(f, " ");
        if (feof(f)) {
            break;
        }
        Other.push_back(ReadSpectrum(f));
    }
    fclose(f);
#else
    ResponseR = SPtr(new Spectrum(*Zero));
    ResponseR->m[0] = _mm_setr_ps(1, 0, 0, 0);
    ResponseG = SPtr(new Spectrum(*Zero));
    ResponseG->m[0] = _mm_setr_ps(0, 1, 0, 0);
    ResponseB = SPtr(new Spectrum(*Zero));
    ResponseB->m[0] = _mm_setr_ps(0, 0, 1, 0);
    Luma = SPtr(new Spectrum(*Zero));
    Luma->m[0] = _mm_setr_ps(0.299f, 0.587f, 0.114f, 0);
    for (int ri = 0; ri <= Depth; ++ri) {
        for (int gi = 0; gi <= Depth - ri; ++gi) {
            int bi = Depth - ri - gi;
            float r = (float)ri / Depth;
            float g = (float)gi / Depth;
            float b = (float)bi / Depth;
            Emission.emplace_back(new Spectrum(*Zero));
            Emission.back()->m[0] = _mm_setr_ps(r, g, b, 0);
            float m = fmaxf(fmaxf(r, g), b);
            r /= m;
            g /= m;
            b /= m;
            Reflection.emplace_back(new Spectrum(*Zero));
            Reflection.back()->m[0] = _mm_setr_ps(r, g, b, 0);
        }
    }
    Emission.shrink_to_fit();
    Reflection.shrink_to_fit();
    Other.push_back(MakeConstantSpectrum(-10));
#endif
    Initialized = true;
}

int Table::RefIndex(int ri, int gi)
{
    return (2 * Depth + 3 - ri) * ri / 2 + gi;
}

void Table::FindInterpolation(
    float r, float g, float b,
    int& i1, float& w1,
    int& i2, float& w2,
    int& i3, float& w3)
{
    if (r < 0) {
        r = 0;
    }
    if (g < 0) {
        g = 0;
    }
    if (b < 0) {
        b = 0;
    }
    float sum = r + g + b;
    if (sum <= 0) {
        i1 = -1;
        w1 = 0;
        i2 = -1;
        w2 = 0;
        i3 = -1;
        w3 = 0;
        return;
    }
    float scale = Depth / sum;
    r *= scale;
    g *= scale;
    b *= scale;
    int ri = (int)r;
    int gi = (int)g;
    int bi = (int)b;
    switch (ri + gi + bi) {
    case Depth:
        i1 = RefIndex(ri, gi);
        w1 = 1;
        i2 = -1;
        w2 = 0;
        i3 = -1;
        w3 = 0;
        return;
    case Depth - 1:
        i1 = RefIndex(ri + 1, gi);
        w1 = r - ri;
        i2 = RefIndex(ri, gi + 1);
        w2 = g - gi;
        i3 = RefIndex(ri, gi);
        w3 = b - bi;
        return;
    case Depth - 2:
        i1 = RefIndex(ri, gi + 1);
        w1 = ri + 1 - r;
        i2 = RefIndex(ri + 1, gi);
        w2 = gi + 1 - g;
        i3 = RefIndex(ri + 1, gi + 1);
        w3 = bi + 1 - b;
    case Depth - 3:
        i1 = RefIndex(ri + 1, gi + 1);
        w1 = 1;
        i2 = -1;
        w2 = 0;
        i3 = -1;
        w3 = 0;
        return;
    default:
        i1 = -1;
        w1 = 0;
        i2 = -1;
        w2 = 0;
        i3 = -1;
        w3 = 0;
        return;
    }
}

Spectrum Table::Interpolate(
    std::vector<SPtr> const& ref,
    float r, float g, float b,
    float scale)
{
    int i1, i2, i3;
    float w1, w2, w3;
    FindInterpolation(r, g, b, i1, w1, i2, w2, i3, w3);
    Spectrum* s1 = (i1 >= 0 && i1 < (int)ref.size()) ? ref[i1].get() : nullptr;
    Spectrum* s2 = (i2 >= 0 && i2 < (int)ref.size()) ? ref[i2].get() : nullptr;
    Spectrum* s3 = (i3 >= 0 && i3 < (int)ref.size()) ? ref[i3].get() : nullptr;
    Spectrum result;
    if (!s1) {
        for (int i = 0; i < Spectrum::Size; ++i) {
            result.m[i] = _mm_setzero_ps();
        }
    } else if (!s2 || !s3) {
        __m128 mw1 = _mm_set1_ps(w1 * scale);
        for (int i = 0; i < Spectrum::Size; ++i) {
            result.m[i] = _mm_mul_ps(mw1, s1->m[i]);
        }
    } else {
        __m128 mw1 = _mm_set1_ps(w1 * scale);
        __m128 mw2 = _mm_set1_ps(w2 * scale);
        __m128 mw3 = _mm_set1_ps(w3 * scale);
        for (int i = 0; i < Spectrum::Size; ++i) {
            __m128 t1 = _mm_mul_ps(mw1, s1->m[i]);
            __m128 t2 = _mm_mul_ps(mw2, s2->m[i]);
            __m128 t3 = _mm_mul_ps(mw3, s3->m[i]);
            result.m[i] = _mm_add_ps(_mm_add_ps(t1, t2), t3);
        }
    }
    return result;
}

Spectrum Spectrum::EmissionSpectrum(FDisp const& c)
{
    Table::Initialize();
    float r, g, b;
    c.unpack(r, g, b);
    return Table::Interpolate(Table::Emission, r, g, b, r + g + b);
}

Spectrum Spectrum::ReflectionSpectrum(FDisp const& c)
{
    Table::Initialize();
    float r, g, b;
    c.unpack(r, g, b);
    return Table::Interpolate(Table::Reflection, r, g, b, fmaxf(fmaxf(r, g), b));
}

Spectrum const& Spectrum::ResponseSpectrumR()
{
    Table::Initialize();
    return *Table::ResponseR;
}

Spectrum const& Spectrum::ResponseSpectrumG()
{
    Table::Initialize();
    return *Table::ResponseG;
}

Spectrum const& Spectrum::ResponseSpectrumB()
{
    Table::Initialize();
    return *Table::ResponseB;
}

Spectrum const& Spectrum::LumaSpectrum()
{
    Table::Initialize();
    return *Table::Luma;
}

Spectrum const& Spectrum::TableSpectrum(int index)
{
    Table::Initialize();
    return *Table::Other.at(index);
}

int Spectrum::TableSize()
{
    Table::Initialize();
    return (int)Table::Other.size();
}

Spectrum const& Spectrum::Zero()
{
    Table::Initialize();
    return *Table::Zero;
}

Spectrum const& Spectrum::One()
{
    Table::Initialize();
    return *Table::One;
}
