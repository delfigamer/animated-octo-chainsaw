#pragma once

#include <immintrin.h>
#include <stdint.h>

struct alignas(32) YmmFloatData {
    float x[8];
};

struct alignas(16) XmmFloatData {
    float x[4];
};

struct alignas(32) YmmIntData {
    int x[8];
};

struct YmmFloat {
    union {
        __m256 m;
        float x[8];
    };

    YmmFloat() = default;
    YmmFloat(__m256 m): m(m) {}
    explicit YmmFloat(__m256i m): m(_mm256_castsi256_ps(m)) {}
    explicit YmmFloat(YmmFloatData const& d): m(_mm256_load_ps(d.x)) {}
    YmmFloat(YmmFloat const&) = default;
    YmmFloat& operator=(YmmFloat const&) = default;

    static YmmFloat set(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
        return _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
    }

    static YmmFloat set1(float f) {
        return _mm256_set1_ps(f);
    }

    operator __m256() const {
        return m;
    }

    explicit operator __m256i() const {
        return _mm256_castps_si256(m);
    }

    explicit operator YmmFloatData() const {
        YmmFloatData d;
        _mm256_store_ps(d.x, m);
        return d;
    }

    friend YmmFloat operator-(YmmFloat a) {
        return a ^ YmmFloat(_mm256_set1_epi32(0x80000000));
    }

    friend YmmFloat operator+(YmmFloat a, YmmFloat b) {
        return _mm256_add_ps(a, b);
    }

    friend YmmFloat operator-(YmmFloat a, YmmFloat b) {
        return _mm256_sub_ps(a, b);
    }

    friend YmmFloat operator*(YmmFloat a, YmmFloat b) {
        return _mm256_mul_ps(a, b);
    }

    YmmFloat recip() const {
        return _mm256_rcp_ps(m);
    }

    friend YmmFloat operator&(YmmFloat a, YmmFloat b) {
        return YmmFloat(_mm256_and_ps(a, b));
    }

    friend YmmFloat operator|(YmmFloat a, YmmFloat b) {
        return YmmFloat(_mm256_or_ps(a, b));
    }

    friend YmmFloat operator^(YmmFloat a, YmmFloat b) {
        return YmmFloat(_mm256_xor_ps(a, b));
    }

    friend YmmFloat operator~(YmmFloat a) {
        return a ^ YmmFloat(_mm256_set1_epi32(-1));
    }

    YmmFloatData data() const {
        return (YmmFloatData)*this;
    }

    bool none() const {
        return _mm256_testz_ps(m, m);
    }

    bool any() const {
        return !none();
    }

    friend YmmFloat operator<(YmmFloat a, YmmFloat b) {
        return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
    }

    friend YmmFloat operator>(YmmFloat a, YmmFloat b) {
        return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
    }

    friend YmmFloat operator<=(YmmFloat a, YmmFloat b) {
        return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
    }

    friend YmmFloat operator>=(YmmFloat a, YmmFloat b) {
        return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
    }

    friend YmmFloat operator==(YmmFloat a, YmmFloat b) {
        return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
    }

    friend YmmFloat operator!=(YmmFloat a, YmmFloat b) {
        return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ);
    }

    static void transpose8(YmmFloat const(&input)[8], YmmFloat& out0, YmmFloat& out1, YmmFloat& out2, YmmFloat& out3, YmmFloat& out4, YmmFloat& out5, YmmFloat& out6, YmmFloat& out7) {
        __m256 t0 = _mm256_unpacklo_ps(input[0], input[1]);
        __m256 t1 = _mm256_unpackhi_ps(input[0], input[1]);
        __m256 t2 = _mm256_unpacklo_ps(input[2], input[3]);
        __m256 t3 = _mm256_unpackhi_ps(input[2], input[3]);
        __m256 t4 = _mm256_unpacklo_ps(input[4], input[5]);
        __m256 t5 = _mm256_unpackhi_ps(input[4], input[5]);
        __m256 t6 = _mm256_unpacklo_ps(input[6], input[7]);
        __m256 t7 = _mm256_unpackhi_ps(input[6], input[7]);
        __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
        out0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
        out1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
        out2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
        out3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
        out4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
        out5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
        out6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
        out7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
    }

    static YmmFloat min(YmmFloat a, YmmFloat b) {
        return _mm256_min_ps(a, b);
    }

    static YmmFloat max(YmmFloat a, YmmFloat b) {
        return _mm256_max_ps(a, b);
    }

    static YmmFloat blend(YmmFloat a0, YmmFloat a1, YmmFloat mask) {
        return _mm256_blendv_ps(a0, a1, mask);
    }

    static YmmFloat fmadd(YmmFloat a, YmmFloat b, YmmFloat c) {
        return _mm256_fmadd_ps(a, b, c);
    }

    int movemask() const {
        return _mm256_movemask_ps(m);
    }

    YmmFloat abs() const {
        YmmFloat mask = YmmFloat(_mm256_set1_epi32(0x7fffffff));
        return mask & m;
    }
};

struct YmmInt {
    union {
        __m256i m;
        int32_t x[8];
    };

    YmmInt() = default;
    YmmInt(__m256i m): m(m) {}
    explicit YmmInt(__m256 m): m(_mm256_castps_si256(m)) {}
    explicit YmmInt(YmmIntData const& d): m(_mm256_load_si256((__m256i const*)d.x)) {}
    YmmInt(YmmInt const&) = default;
    YmmInt& operator=(YmmInt const&) = default;

    static YmmInt set(int32_t x0, int32_t x1, int32_t x2, int32_t x3, int32_t x4, int32_t x5, int32_t x6, int32_t x7) {
        return _mm256_setr_epi32(x0, x1, x2, x3, x4, x5, x6, x7);
    }

    static YmmInt set1(int32_t x) {
        return _mm256_set1_epi32(x);
    }

    operator __m256i() const {
        return m;
    }

    explicit operator __m256() const {
        return _mm256_castsi256_ps(m);
    }

    explicit operator YmmIntData() const {
        YmmIntData d;
        _mm256_store_si256((__m256i*)d.x, m);
        return d;
    }

    friend YmmInt operator+(YmmInt a, YmmInt b) {
        return _mm256_add_epi32(a, b);
    }

    friend YmmInt operator-(YmmInt a, YmmInt b) {
        return _mm256_sub_epi32(a, b);
    }

    friend YmmInt operator&(YmmInt a, YmmInt b) {
        return _mm256_and_si256(a, b);
    }

    friend YmmInt operator|(YmmInt a, YmmInt b) {
        return _mm256_or_si256(a, b);
    }

    friend YmmInt operator^(YmmInt a, YmmInt b) {
        return _mm256_xor_si256(a, b);
    }

    friend YmmInt operator~(YmmInt a) {
        return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
    }

    friend YmmInt operator<(YmmInt a, YmmInt b) {
        return _mm256_cmpgt_epi32(b, a);
    }

    friend YmmInt operator>(YmmInt a, YmmInt b) {
        return _mm256_cmpgt_epi32(a, b);
    }

    friend YmmInt operator<=(YmmInt a, YmmInt b) {
        return ~(a > b);
    }

    friend YmmInt operator>=(YmmInt a, YmmInt b) {
        return ~(a < b);
    }

    friend YmmInt operator==(YmmInt a, YmmInt b) {
        return _mm256_cmpeq_epi32(a, b);
    }

    friend YmmInt operator!=(YmmInt a, YmmInt b) {
        return ~(a == b);
    }

    YmmIntData data() const {
        return (YmmIntData)*this;
    }

    bool none_bit() const {
        return _mm256_testz_si256(m, m);
    }

    bool any_bit() const {
        return !none_bit();
    }

    int movemask() const {
        return YmmFloat(m).movemask();
    }

    static YmmInt blend(YmmInt a0, YmmInt a1, YmmInt mask) {
        return (a0 & ~mask) | (a1 & mask);
    }
};

struct XmmFloat {
    union {
        __m128 m;
        float x[4];
    };

    XmmFloat() = default;
    XmmFloat(__m128 m): m(m) {}
    explicit XmmFloat(XmmFloatData const& d): m(_mm_load_ps(d.x)) {}
    XmmFloat(XmmFloat const&) = default;
    XmmFloat& operator=(XmmFloat const&) = default;

    operator __m128() const {
        return m;
    }

    static void transpose8x4(XmmFloat(&input)[8], YmmFloat& out0, YmmFloat& out1, YmmFloat& out2, YmmFloat& out3) {
        __m256 row0 = _mm256_setr_m128(input[0], input[4]);
        __m256 row1 = _mm256_setr_m128(input[1], input[5]);
        __m256 row2 = _mm256_setr_m128(input[2], input[6]);
        __m256 row3 = _mm256_setr_m128(input[3], input[7]);
        __m256 tmp0 = _mm256_shuffle_ps(row0, row1, 0x44);
        __m256 tmp1 = _mm256_shuffle_ps(row2, row3, 0x44);
        __m256 tmp2 = _mm256_shuffle_ps(row0, row1, 0xEE);
        __m256 tmp3 = _mm256_shuffle_ps(row2, row3, 0xEE);
        out0 = _mm256_shuffle_ps(tmp0, tmp1, 0x88);
        out1 = _mm256_shuffle_ps(tmp0, tmp1, 0xDD);
        out2 = _mm256_shuffle_ps(tmp2, tmp3, 0x88);
        out3 = _mm256_shuffle_ps(tmp2, tmp3, 0xDD);
    }
};
