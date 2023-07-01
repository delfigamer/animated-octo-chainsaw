#pragma once

#include "FLin.h"
#include <xmmintrin.h>

struct Spectrum
{
    static constexpr int Size = 100;
    __m128 m[Size];

    Spectrum();
    ~Spectrum();

    static Spectrum EmissionSpectrum(FDisp const& c);
    static Spectrum ReflectionSpectrum(FDisp const& c);
    static Spectrum const& ResponseSpectrumR();
    static Spectrum const& ResponseSpectrumG();
    static Spectrum const& ResponseSpectrumB();
    static Spectrum const& LumaSpectrum();
    static Spectrum const& TableSpectrum(int index);
    static int TableSize();
    static Spectrum const& Zero();
    static Spectrum const& One();

    float mean() const
    {
        __m128 acc = _mm_setzero_ps();
        for (int i = 0; i < Spectrum::Size; ++i) {
            acc = _mm_add_ps(acc, m[i]);
        }
        alignas(16) float buf[4];
        _mm_store_ps(buf, acc);
        return (0.25f / Size) * (buf[0] + buf[1] + buf[2] + buf[3]);
    }

    Spectrum exp2() const
    {
        Spectrum ret;
        for (int i = 0; i < Spectrum::Size; ++i) {
            ret.m[i] = _mm_exp2_ps(m[i]);
        }
        return ret;
    }

    FDisp Responce() const;
};

inline Spectrum::Spectrum()
{
}

inline Spectrum::~Spectrum()
{
}

inline float dot(Spectrum const& a, Spectrum const& b)
{
    __m128 acc = _mm_setzero_ps();
    for (int i = 0; i < Spectrum::Size; ++i) {
        acc = _mm_add_ps(acc, _mm_mul_ps(a.m[i], b.m[i]));
    }
    alignas(16) float buf[4];
    _mm_store_ps(buf, acc);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

inline float luma(Spectrum const& a)
{
    return dot(a, Spectrum::LumaSpectrum());
}

inline void modulate(Spectrum& a, Spectrum const& b, float c)
{
    __m128 mc = _mm_set1_ps(c);
    for (int i = 0; i < Spectrum::Size; ++i) {
        a.m[i] = _mm_mul_ps(a.m[i], _mm_mul_ps(b.m[i], mc));
    }
}

inline void modulate(Spectrum& a, Spectrum const& b)
{
    for (int i = 0; i < Spectrum::Size; ++i) {
        a.m[i] = _mm_mul_ps(a.m[i], b.m[i]);
    }
}

inline void modulate(Spectrum& a, float c)
{
    __m128 mc = _mm_set1_ps(c);
    for (int i = 0; i < Spectrum::Size; ++i) {
        a.m[i] = _mm_mul_ps(a.m[i], mc);
    }
}

inline void fmadd(Spectrum& a, float b, float c)
{
    __m128 mb = _mm_set1_ps(b);
    __m128 mc = _mm_set1_ps(c);
    for (int i = 0; i < Spectrum::Size; ++i) {
        a.m[i] = _mm_add_ps(_mm_mul_ps(a.m[i], mb), mc);
    }
}

inline FDisp Spectrum::Responce() const
{
    float r, g, b;
    r = dot(*this, ResponseSpectrumR());
    g = dot(*this, ResponseSpectrumG());
    b = dot(*this, ResponseSpectrumB());
    return FDisp{ r, g, b };
}
