#include "Random.h"
#include <stdexcept>

template<typename T>
static T ceil_log2(T x) {
    if (x > ((T)(-1) >> 1)) {
        throw std::runtime_error("rounding up to a power of 2 would result in overflow");
    }
    T r = 1;
    while (r < x) {
        r <<= 1;
    }
    return r;
}

static uint32_t rotr32(uint32_t x, unsigned r) {
    return x >> r | x << ((32 - r) & 31);
}

static constexpr uint64_t multiplier = 6364136223846793005u;
static constexpr uint64_t increment = 1442695040888963407u;

Random::Random() {
    state = 0x4d595df4d0f33173u;
}
 
void Random::uniform(uint32_t& r) {
    uint64_t x = state;
    unsigned count = (unsigned)(x >> 59);
    state = x * multiplier + increment;
    x ^= x >> 18;
    r = rotr32((uint32_t)(x >> 27), count);
}

void Random::uniform_below(uint32_t limit, uint32_t& r) {
    uint32_t mask = ceil_log2(limit) - 1;
    do {
        uniform(r);
        r &= mask;
    } while (r >= limit);
}

void Random::uniform(float& q) {
    uint32_t u;
    uniform(u);
    q = 0x1p-32f * (float)u;
}

void Random::triangle(float& u) {
    float q1, q2;
    uniform(q1);
    uniform(q2);
    u = q1 - q2;
}

void Random::circle(float& u, float& v) {
    while (true) {
        uniform(u);
        u = u * 2.0f - 1.0f;
        uniform(v);
        v = v * 2.0f - 1.0f;
        float uvsqr = u * u + v * v;
        if (uvsqr > 1.0f) {
            continue;
        }
        float inv = 1.0f / sqrtf(uvsqr);
        u *= inv;
        v *= inv;
        return;
    }
}

void Random::lambert(Vec3 const& n, Vec3& d) {
    Vec3 a{1, 0, 0};
    if (n.x > 0.8f || n.x < -0.8f) {
        a = Vec3{0, 1, 0};
    }
    Vec3 b = norm(cross(a, n));
    Vec3 t = norm(cross(b, n));
    float q;
    uniform(q);
    float z = sqrtf(q);
    float r = sqrtf(1.0f - q);
    float u;
    float v;
    circle(u, v);
    u *= r;
    v *= r;
    d = z * n + u * b + v * t;
}

void Random::p(float beta, bool& r) {
    float q;
    uniform(q);
    r = q < beta;
}

DiscreteDistribution DiscreteDistribution::make(std::vector<float> const& input_weights) {
    constexpr uint64_t cp_unit = (uint64_t)1 << 32;
    size_t size = ceil_log2(input_weights.size());
    float total_weight = 0.0f;
    for (size_t i = 0; i < input_weights.size(); ++i) {
        total_weight += input_weights[i];
    }
    std::vector<uint64_t> scaled_commit_probabilities(size);
    std::vector<size_t> overfull_indices;
    std::vector<size_t> underfull_indices;
    float probability_scale = (float)cp_unit * (float)size / total_weight;
    for (size_t i = 0; i < input_weights.size(); ++i) {
        scaled_commit_probabilities[i] = probability_scale * input_weights[i];
        if (scaled_commit_probabilities[i] < cp_unit) {
            underfull_indices.push_back(i);
        } else {
            if (scaled_commit_probabilities[i] > cp_unit) {
                overfull_indices.push_back(i);
            }
        }
    }
    for (size_t i = input_weights.size(); i < size; ++i) {
        scaled_commit_probabilities[i] = 0;
        underfull_indices.push_back(i);
    }
    std::vector<Elem> elems(size);
    for (size_t i = 0; i < size; ++i) {
        elems[i].alt = i;
        elems[i].threshold = 0xffffffff;
    }
    while (!overfull_indices.empty() && !underfull_indices.empty()) {
        size_t next_overfull = overfull_indices.back();
        size_t next_underfull = underfull_indices.back();
        underfull_indices.pop_back();
        uint64_t delta = cp_unit - scaled_commit_probabilities[next_underfull];
        scaled_commit_probabilities[next_overfull] -= delta;
        elems[next_underfull].alt = next_overfull;
        elems[next_underfull].threshold = cp_unit - delta;
        if (scaled_commit_probabilities[next_overfull] <= cp_unit) {
            overfull_indices.pop_back();
            if (scaled_commit_probabilities[next_overfull] < cp_unit) {
                underfull_indices.push_back(next_overfull);
            }
        }
    }
    std::vector<float> probs(size, 0.0f);
    for (size_t i = 0; i < size; ++i) {
        probs[i] += elems[i].threshold;
        probs[elems[i].alt] += cp_unit - elems[i].threshold;
    }
    for (size_t i = 0; i < size; ++i) {
        probs[i] *= 1.0f / (size * cp_unit);
    }
    DiscreteDistribution dd;
    dd.elems = std::move(elems);
    dd.probs = std::move(probs);
    return dd;
}

void DiscreteDistribution::sample(Random& random, size_t& result) {
    uint32_t qi, qa;
    random.uniform(qi);
    result = qi & (elems.size() - 1);
    random.uniform(qa);
    if (qa >= elems[result].threshold) {
        result = elems[result].alt;
    }
}

float DiscreteDistribution::probability_of(size_t i) {
    return probs[i];
}
