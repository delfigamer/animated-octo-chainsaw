#pragma once

#include <string>
#include <vector>
#include <math.h>

struct Vec3 {
    float x, y, z;

    float& operator[](size_t i) {
        return ((float*)this)[i];
    }

    float operator[](size_t i) const {
        return ((float*)this)[i];
    }
};

struct Xform3 {
    Vec3 offset, xc, yc, zc;
};

inline Vec3 operator+(Vec3 a, Vec3 b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(Vec3 a, Vec3 b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(float a, Vec3 b) {
    return Vec3{a * b.x, a * b.y, a * b.z};
}

inline Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float dotsqr(Vec3 a) {
    return dot(a, a);
}

inline Vec3 norm(Vec3 a) {
    return (1.0f / sqrtf(dotsqr(a))) * a;
}

inline Vec3 elementwise_inverse(Vec3 a) {
    return Vec3{1.0f / a.x, 1.0f / a.y, 1.0f / a.z};
}

inline float det3(Vec3 a, Vec3 b, Vec3 c) {
    return dot(cross(a, b), c);
}
