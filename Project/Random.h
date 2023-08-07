#pragma once

#include <stdint.h>
#include <vector>
#include "Vec3.h"

class Random {
private:
    uint64_t state;

public:
    Random();
    Random(Random const&) = default;
    Random& operator=(Random const&) = default;
    ~Random() = default;

    void uniform(uint32_t& r);
    void uniform_below(uint32_t limit, uint32_t& r);
    void uniform(float& q);
    void triangle(float& u);
    void circle(float& u, float& v);
    void lambert(Vec3 const& n, Vec3& d);
    void p(float beta, bool& r);
};

class DiscreteDistribution {
private:
    struct Elem {
        size_t alt;
        uint32_t threshold;
    };

    std::vector<Elem> elems;
    std::vector<float> probs;

public:
    DiscreteDistribution() = default;
    DiscreteDistribution(DiscreteDistribution&& other) = default;
    DiscreteDistribution(DiscreteDistribution const& other) = default;
    DiscreteDistribution& operator=(DiscreteDistribution&& other) = default;
    DiscreteDistribution& operator=(DiscreteDistribution const& other) = default;
    ~DiscreteDistribution() = default;

    static DiscreteDistribution make(std::vector<float> const& input_weights);
    void sample(Random& random, size_t& result);
    float probability_of(size_t i);
};
