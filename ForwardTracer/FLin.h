#pragma once

#include <cmath>
#include <xmmintrin.h>
#include <pmmintrin.h>

namespace
{
    inline __m128 _mm_exp2_ps(__m128 x)
    {
        __m128 f, g, e, p, r;
        __m128i i, j;
        __m128 b = _mm_set1_ps(-80);
        __m128 c0 = _mm_set1_ps(0.3371894346f);
        __m128 c1 = _mm_set1_ps(0.657636276f);
        __m128 c2 = _mm_set1_ps(1.00172476f);
        /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
        g = _mm_cmpgt_ps(x, b);             /* g = x > b */
        i = _mm_cvttps_epi32(x);            /* i = (int)t */
        j = _mm_srli_epi32(_mm_castps_si128(x), 31); /* signbit(t) */
        i = _mm_sub_epi32(i, j);            /* (int)t - signbit(t) */
        e = _mm_cvtepi32_ps(i);             /* floor(t) ~= (int)t - signbit(t) */
        f = _mm_sub_ps(x, e);               /* f = t - floor(t) */
        p = c0;                             /* c0 */
        p = _mm_mul_ps(p, f);               /* c0 * f */
        p = _mm_add_ps(p, c1);              /* c0 * f + c1 */
        p = _mm_mul_ps(p, f);               /* (c0 * f + c1) * f */
        p = _mm_add_ps(p, c2);              /* p = (c0 * f + c1) * f + c2 ~= 2^f */
        j = _mm_slli_epi32(i, 23);          /* i << 23 */
        r = _mm_castsi128_ps(_mm_add_epi32(j, _mm_castps_si128(p))); /* r = p * 2^i*/
        r = _mm_and_ps(r, g);               /* r = x > b ? r : 0 */
        return r;
    }
}

struct FXmmQuad
{
    __m128 m;

    FXmmQuad() = default;

    explicit FXmmQuad(__m128 mm)
        : m(mm)
    {
    }

    FXmmQuad(float x, float y, float z, float w)
    {
        m = _mm_setr_ps(x, y, z, w);
    }

    void unpack(float& x, float& y, float& z, float& w) const
    {
        alignas(16) float buf[4];
        _mm_store_ps(buf, m);
        x = buf[0];
        y = buf[1];
        z = buf[2];
        w = buf[3];
    }

    void unpack(float& x, float& y, float& z) const
    {
        float dummy;
        unpack(x, y, z, dummy);
    }

    float x() const
    {
        return _mm_cvtss_f32(m);
    }
};

struct FPoint: FXmmQuad
{
    FPoint() = default;

    explicit FPoint(__m128 mm)
        : FXmmQuad(mm)
    {
    }

    FPoint(float x, float y, float z)
        : FXmmQuad(x, y, z, 1)
    {
    }
};

struct FDisp: FXmmQuad
{
    FDisp() = default;

    explicit FDisp(__m128 mm)
        : FXmmQuad(mm)
    {
    }

    FDisp(float x, float y, float z)
        : FXmmQuad(x, y, z, 0)
    {
    }

    explicit FDisp(FXmmQuad q)
    {
        m = _mm_mul_ps(q.m, _mm_setr_ps(1, 1, 1, 0));
    }
};

struct FMat
{
    __m128 ma;
    __m128 mb;
    __m128 mc;

    FMat() = default;

    FMat(float xx, float xy, float xz, float xw, float yx, float yy, float yz, float yw, float zx, float zy, float zz, float zw)
    {
        alignas(16) float buf[12] = {
            xx, xy, xz, xw,
            yx, yy, yz, yw,
            zx, zy, zz, zw };
        assemble(buf);
    }

    FMat(float xx, float xy, float xz, float yx, float yy, float yz, float zx, float zy, float zz)
    {
        alignas(16) float buf[12] = {
            xx, xy, xz, 0,
            yx, yy, yz, 0,
            zx, zy, zz, 0 };
        assemble(buf);
    }

    static FMat basis(FDisp const& x, FDisp const& y, FDisp const& z, FPoint const& w)
    {
        FMat r;
        __m128 ta = _mm_unpacklo_ps(x.m, y.m);
        __m128 tb = _mm_unpacklo_ps(z.m, w.m);
        __m128 tc = _mm_unpackhi_ps(x.m, y.m);
        __m128 td = _mm_unpackhi_ps(z.m, w.m);
        r.ma = _mm_movelh_ps(ta, tb);
        r.mb = _mm_movehl_ps(tb, ta);
        r.mc = _mm_movelh_ps(tc, td);
        return r;
    }

    static FMat identity()
    {
        FMat r;
        r.ma = _mm_setr_ps(1, 0, 0, 0);
        r.mb = _mm_setr_ps(0, 1, 0, 0);
        r.mc = _mm_setr_ps(0, 0, 1, 0);
        return r;
    }

    void unpackptr(FDisp* xp, FDisp* yp, FDisp* zp, FPoint* wp) const
    {
        __m128 md = _mm_setr_ps(0, 0, 0, 1);
        if (xp || yp) {
            __m128 ta = _mm_unpacklo_ps(ma, mb);
            __m128 tb = _mm_unpacklo_ps(mc, md);
            if (xp) {
                xp->m = _mm_movelh_ps(ta, tb);
            }
            if (yp) {
                yp->m = _mm_movehl_ps(tb, ta);
            }
        }
        if (zp || wp) {
            __m128 tc = _mm_unpackhi_ps(ma, mb);
            __m128 td = _mm_unpackhi_ps(mc, md);
            if (zp) {
                zp->m = _mm_movelh_ps(tc, td);
            }
            if (wp) {
                wp->m = _mm_movehl_ps(td, tc);
            }
        }
    }

    FXmmQuad xdual() const
    {
        return FXmmQuad{ ma };
    }

    FXmmQuad ydual() const
    {
        return FXmmQuad{ mb };
    }

    FXmmQuad zdual() const
    {
        return FXmmQuad{ mc };
    }

    FDisp xunit() const
    {
        FDisp r;
        unpackptr(&r, nullptr, nullptr, nullptr);
        return r;
    }

    FDisp yunit() const
    {
        FDisp r;
        unpackptr(nullptr, &r, nullptr, nullptr);
        return r;
    }

    FDisp zunit() const
    {
        FDisp r;
        unpackptr(nullptr, nullptr, &r, nullptr);
        return r;
    }

    FPoint origin() const
    {
        FPoint r;
        unpackptr(nullptr, nullptr, nullptr, &r);
        return r;
    }

    void disassemble(float* f) const
    {
        _mm_storeu_ps(f + 0, ma);
        _mm_storeu_ps(f + 4, mb);
        _mm_storeu_ps(f + 8, mc);
    }

    static FMat assemble(float const* f)
    {
        FMat r;
        r.ma = _mm_loadu_ps(f + 0);
        r.mb = _mm_loadu_ps(f + 4);
        r.mc = _mm_loadu_ps(f + 8);
        return r;
    }
};

inline FDisp operator*(float s, FDisp const& d)
{
    return FDisp{ _mm_mul_ps(_mm_set_ps1(s), d.m) };
}

inline FDisp operator*(FDisp const& d, float s)
{
    return FDisp{ _mm_mul_ps(_mm_set_ps1(s), d.m) };
}

inline FDisp operator*(FDisp const& a, FDisp const& b)
{
    return FDisp{ _mm_mul_ps(a.m, b.m) };
}

inline FDisp operator/(FDisp const& d, float s)
{
    return FDisp{ _mm_div_ps(d.m, _mm_set_ps1(s)) };
}

inline FDisp operator+(FDisp const& a, FDisp const& b)
{
    return FDisp{ _mm_add_ps(a.m, b.m) };
}

inline FDisp operator-(FDisp const& a, FDisp const& b)
{
    return FDisp{ _mm_sub_ps(a.m, b.m) };
}

inline FPoint operator+(FPoint const& a, FDisp const& b)
{
    return FPoint{ _mm_add_ps(a.m, b.m) };
}

inline FPoint operator+(FDisp const& a, FPoint const& b)
{
    return FPoint{ _mm_add_ps(a.m, b.m) };
}

inline FPoint operator-(FPoint const& a, FDisp const& b)
{
    return FPoint{ _mm_sub_ps(a.m, b.m) };
}

inline FDisp operator-(FPoint const& a, FPoint const& b)
{
    return FDisp{ _mm_sub_ps(a.m, b.m) };
}

inline FDisp operator-(FDisp const& b)
{
    return FDisp{ _mm_sub_ps(_mm_setzero_ps(), b.m) };
}

inline FDisp& operator*=(FDisp& d, float s)
{
    d.m = _mm_mul_ps(_mm_set_ps1(s), d.m);
    return d;
}

inline FDisp& operator*=(FDisp& a, FDisp const& b)
{
    a.m = _mm_mul_ps(a.m, b.m);
    return a;
}

inline FDisp& operator/=(FDisp& d, float s)
{
    d.m = _mm_div_ps(d.m, _mm_set_ps1(s));
    return d;
}

inline FDisp& operator+=(FDisp& a, FDisp const& b)
{
    a.m = _mm_add_ps(a.m, b.m);
    return a;
}

inline FDisp& operator-=(FDisp& a, FDisp const& b)
{
    a.m = _mm_sub_ps(a.m, b.m);
    return a;
}

inline FPoint& operator+=(FPoint& a, FDisp const& b)
{
    a.m = _mm_add_ps(a.m, b.m);
    return a;
}

inline FPoint& operator-=(FPoint& a, FDisp const& b)
{
    a.m = _mm_sub_ps(a.m, b.m);
    return a;
}

inline FDisp cross(FDisp const& a, FDisp const& b)
{
    __m128 ayzx = _mm_shuffle_ps(a.m, a.m, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 azxy = _mm_shuffle_ps(a.m, a.m, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 byzx = _mm_shuffle_ps(b.m, b.m, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 bzxy = _mm_shuffle_ps(b.m, b.m, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 t = _mm_mul_ps(ayzx, bzxy);
    __m128 u = _mm_mul_ps(azxy, byzx);
    return FDisp{ _mm_sub_ps(t, u) };
}

inline float dot(FXmmQuad const& a, FXmmQuad const& b)
{
    __m128 ab = _mm_mul_ps(a.m, b.m);
    alignas(16) float buf[4];
    _mm_store_ps(buf, ab);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

inline FDisp norm(FDisp const& a)
{
    __m128 asqr = _mm_mul_ps(a.m, a.m);
    __m128 t1 = _mm_hadd_ps(asqr, asqr);
    __m128 t2 = _mm_hadd_ps(t1, t1);
    __m128 scale = _mm_rsqrt_ps(t2);
    return FDisp{ _mm_mul_ps(a.m, scale) };
}

inline FDisp reflection(FDisp const& n, FDisp const& a)
{
    __m128 nx = _mm_shuffle_ps(n.m, n.m, _MM_SHUFFLE(3, 0, 0, 0));
    __m128 ny = _mm_shuffle_ps(n.m, n.m, _MM_SHUFFLE(3, 1, 1, 1));
    __m128 nz = _mm_shuffle_ps(n.m, n.m, _MM_SHUFFLE(3, 2, 2, 2));
    __m128 n2 = _mm_mul_ps(n.m, _mm_setr_ps(2, 2, 2, 2));
    __m128 an = _mm_mul_ps(a.m, n2);
    __m128 anx = _mm_mul_ps(an, nx);
    __m128 any = _mm_mul_ps(an, ny);
    __m128 anz = _mm_mul_ps(an, nz);
    __m128 ann = _mm_add_ps(anx, _mm_add_ps(any, anz));
    return FDisp{ _mm_sub_ps(ann, a.m) };
}

inline float luma(FDisp const& a)
{
    static FDisp wp{ 0.299f, 0.587f, 0.114f };
    return dot(a, wp);
}

inline FMat operator+(FMat const& a, FMat const& b)
{
    FMat r;
    r.ma = _mm_add_ps(a.ma, b.ma);
    r.mb = _mm_add_ps(a.mb, b.mb);
    r.mc = _mm_add_ps(a.mc, b.mc);
    return r;
}

inline FMat operator-(FMat const& a, FMat const& b)
{
    FMat r;
    r.ma = _mm_sub_ps(a.ma, b.ma);
    r.mb = _mm_sub_ps(a.mb, b.mb);
    r.mc = _mm_sub_ps(a.mc, b.mc);
    return r;
}

inline FPoint operator*(FMat const& m, FPoint const& p)
{
    __m128 a = _mm_mul_ps(m.ma, p.m);
    __m128 b = _mm_mul_ps(m.mb, p.m);
    __m128 c = _mm_mul_ps(m.mc, p.m);
    __m128 d = _mm_setr_ps(0, 0, 0, 1);
    __m128 abl = _mm_unpacklo_ps(a, b);
    __m128 abh = _mm_unpackhi_ps(a, b);
    __m128 cdl = _mm_unpacklo_ps(c, d);
    __m128 cdh = _mm_unpackhi_ps(c, d);
    __m128 ab = _mm_add_ps(abl, abh);
    __m128 cd = _mm_add_ps(cdl, cdh);
    __m128 t = _mm_movelh_ps(ab, cd);
    __m128 u = _mm_movehl_ps(cd, ab);
    __m128 r = _mm_add_ps(t, u);
    return FPoint{ r };
}

inline FDisp operator*(FMat const& m, FDisp const& p)
{
    __m128 a = _mm_mul_ps(m.ma, p.m);
    __m128 b = _mm_mul_ps(m.mb, p.m);
    __m128 c = _mm_mul_ps(m.mc, p.m);
    __m128 d = _mm_setzero_ps();
    __m128 abl = _mm_unpacklo_ps(a, b);
    __m128 abh = _mm_unpackhi_ps(a, b);
    __m128 cdl = _mm_unpacklo_ps(c, d);
    __m128 cdh = _mm_unpackhi_ps(c, d);
    __m128 ab = _mm_add_ps(abl, abh);
    __m128 cd = _mm_add_ps(cdl, cdh);
    __m128 t = _mm_movelh_ps(ab, cd);
    __m128 u = _mm_movehl_ps(cd, ab);
    __m128 r = _mm_add_ps(t, u);
    return FDisp{ r };
}

inline float sqr(FMat const& m)
{
    __m128 a = _mm_mul_ps(m.ma, m.ma);
    __m128 b = _mm_mul_ps(m.mb, m.mb);
    __m128 c = _mm_mul_ps(m.mc, m.mc);
    a = _mm_add_ps(a, b);
    a = _mm_add_ps(a, c);
    alignas(16) float buf[4];
    _mm_store_ps(buf, a);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

// Alright, these ops happen so rarely it's absolutely not worth the effort
inline FMat operator*(FMat const& a, FMat const& b)
{
    float ac[12];
    float bc[12];
    float rc[12];
    a.disassemble(ac);
    b.disassemble(bc);
    rc[0] = ac[0] * bc[0] + ac[1] * bc[4] + ac[2] * bc[8];
    rc[1] = ac[0] * bc[1] + ac[1] * bc[5] + ac[2] * bc[9];
    rc[2] = ac[0] * bc[2] + ac[1] * bc[6] + ac[2] * bc[10];
    rc[3] = ac[0] * bc[3] + ac[1] * bc[7] + ac[2] * bc[11] + ac[3];
    rc[4] = ac[4] * bc[0] + ac[5] * bc[4] + ac[6] * bc[8];
    rc[5] = ac[4] * bc[1] + ac[5] * bc[5] + ac[6] * bc[9];
    rc[6] = ac[4] * bc[2] + ac[5] * bc[6] + ac[6] * bc[10];
    rc[7] = ac[4] * bc[3] + ac[5] * bc[7] + ac[6] * bc[11] + ac[7];
    rc[8] = ac[8] * bc[0] + ac[9] * bc[4] + ac[10] * bc[8];
    rc[9] = ac[8] * bc[1] + ac[9] * bc[5] + ac[10] * bc[9];
    rc[10] = ac[8] * bc[2] + ac[9] * bc[6] + ac[10] * bc[10];
    rc[11] = ac[8] * bc[3] + ac[9] * bc[7] + ac[10] * bc[11] + ac[11];
    return FMat::assemble(rc);
}

inline FMat inverse(FMat const& m)
{
    float mc[12];
    float rc[12];
    m.disassemble(mc);
    float det =
        mc[0] * mc[5] * mc[10]
        + mc[1] * mc[6] * mc[8]
        + mc[2] * mc[4] * mc[9]
        - mc[0] * mc[6] * mc[9]
        - mc[1] * mc[4] * mc[10]
        - mc[2] * mc[5] * mc[8];
    if (fabsf(det) < 1e-9) {
        return FMat{};
    }
    float invdet = 1.0f / det;
    rc[0] = invdet * (mc[5] * mc[10] - mc[6] * mc[9]);
    rc[1] = invdet * (mc[2] * mc[9] - mc[1] * mc[10]);
    rc[2] = invdet * (mc[1] * mc[6] - mc[2] * mc[5]);
    rc[3] = invdet * (
        mc[1] * mc[7] * mc[10] + mc[2] * mc[5] * mc[11] + mc[3] * mc[6] * mc[9]
        - mc[1] * mc[6] * mc[11] - mc[2] * mc[7] * mc[9] - mc[3] * mc[5] * mc[10]);
    rc[4] = invdet * (mc[6] * mc[8] - mc[4] * mc[10]);
    rc[5] = invdet * (mc[0] * mc[10] - mc[2] * mc[8]);
    rc[6] = invdet * (mc[2] * mc[4] - mc[0] * mc[6]);
    rc[7] = invdet * (
        mc[0] * mc[6] * mc[11] + mc[2] * mc[7] * mc[8] + mc[3] * mc[4] * mc[10]
        - mc[0] * mc[7] * mc[10] - mc[2] * mc[4] * mc[11] - mc[3] * mc[6] * mc[8]);
    rc[8] = invdet * (mc[4] * mc[9] - mc[5] * mc[8]);
    rc[9] = invdet * (mc[1] * mc[8] - mc[0] * mc[9]);
    rc[10] = invdet * (mc[0] * mc[5] - mc[1] * mc[4]);
    rc[11] = invdet * (
        mc[0] * mc[7] * mc[9] + mc[1] * mc[4] * mc[11] + mc[3] * mc[5] * mc[8]
        - mc[0] * mc[5] * mc[11] - mc[1] * mc[7] * mc[8] - mc[3] * mc[4] * mc[9]);
    return FMat::assemble(rc);
}
