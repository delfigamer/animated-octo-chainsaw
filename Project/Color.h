#pragma once

#include <array>
#include <math.h>

struct Color: std::array<float, 3> {
    template<typename F>
    friend Color elementwise_map(F f, Color a) {
        Color r;
        for (size_t i = 0; i < a.size(); ++i) {
            r[i] = f(a[i]);
        }
        return r;
    }

    template<typename F>
    friend Color& elementwise_map_inplace(F f, Color& a) {
        for (size_t i = 0; i < a.size(); ++i) {
            f(a[i]);
        }
        return a;
    }

    template<typename F>
    friend Color elementwise_zip(F f, Color a, Color b) {
        Color r;
        for (size_t i = 0; i < a.size(); ++i) {
            r[i] = f(a[i], b[i]);
        }
        return r;
    }

    template<typename F>
    friend Color& elementwise_zip_inplace(F f, Color& a, Color b) {
        for (size_t i = 0; i < a.size(); ++i) {
            f(a[i], b[i]);
        }
        return a;
    }

    template<typename F>
    friend float elementwise_foldl1(F f, Color a) {
        float x = a[0];
        for (size_t i = 1; i < a.size(); ++i) {
            x = f(x, a[i]);
        }
        return x;
    }

    Color& operator+=(Color other) {
        return elementwise_zip_inplace([](float& x, float y) { x += y; }, *this, other);
    }

    Color& operator*=(Color other) {
        return elementwise_zip_inplace([](float& x, float y) { x *= y; }, *this, other);
    }

    friend Color operator+(Color a, Color b) {
        return elementwise_zip([](float x, float y) { return x + y; }, a, b);
    }

    friend Color operator*(float a, Color b) {
        return elementwise_map([=](float y) { return a * y; }, b);
    }

    friend Color operator*(Color a, Color b) {
        return elementwise_zip([=](float x, float y) { return x * y; }, a, b);
    }
};

//struct Xform3 {
//    Vec3 offset, xc, yc, zc;
//};
//
//inline bool operator==(Vec3 a, Vec3 b) {
//    return a.x == b.x && a.y == b.y && a.z == b.z;
//}
//
//inline Vec3 operator-(Vec3 b) {
//    return Vec3{-b.x, -b.y, -b.z};
//}
//
//inline Vec3 operator+(Vec3 a, Vec3 b) {
//    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
//}
//
//inline Vec3 operator-(Vec3 a, Vec3 b) {
//    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
//}
//
//inline Vec3 operator*(float a, Vec3 b) {
//    return Vec3{a * b.x, a * b.y, a * b.z};
//}
//
//inline Vec3 cross(Vec3 a, Vec3 b) {
//    return Vec3{
//        a.y * b.z - a.z * b.y,
//        a.z * b.x - a.x * b.z,
//        a.x * b.y - a.y * b.x,
//    };
//}
//
//inline float dot(Vec3 a, Vec3 b) {
//    return a.x * b.x + a.y * b.y + a.z * b.z;
//}
//
//inline float dotsqr(Vec3 a) {
//    return dot(a, a);
//}
//
//inline float length(Vec3 a) {
//    return sqrtf(dotsqr(a));
//}
//
//inline Vec3 norm(Vec3 a) {
//    return (1.0f / sqrtf(dotsqr(a))) * a;
//}
//
//inline void assign_elementwise_product(Vec3& a, Vec3 b) {
//    a.x *= b.x;
//    a.y *= b.y;
//    a.z *= b.z;
//}
//
//inline Vec3 elementwise_product(Vec3 a, Vec3 b) {
//    return Vec3{a.x * b.x, a.y * b.y, a.z * b.z};
//}
//
//inline Vec3 elementwise_inverse(Vec3 a) {
//    return Vec3{1.0f / a.x, 1.0f / a.y, 1.0f / a.z};
//}
//
//inline float det3(Vec3 a, Vec3 b, Vec3 c) {
//    return dot(cross(a, b), c);
//}
