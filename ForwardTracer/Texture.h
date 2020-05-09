#pragma once

#include "FLin.h"
#include <vector>
#include <cstdint>

class ColorTexture
{
private:
    int width;
    int height;
    std::vector<uint8_t> pixels;

public:
    ColorTexture();
    ~ColorTexture();
    ColorTexture(ColorTexture const& other);
    ColorTexture(ColorTexture&& other);
    ColorTexture& operator=(ColorTexture const& other);
    ColorTexture& operator=(ColorTexture&& other);

    void Load(char const* path);
    FDisp Pixel(int ix, int iy) const;
    FDisp Sample(float u, float v) const;
    bool Empty() const;
};

class ParameterTexture
{
private:
    int width;
    int height;
    std::vector<uint8_t> pixels;

public:
    ParameterTexture();
    ~ParameterTexture();
    ParameterTexture(ParameterTexture const& other);
    ParameterTexture(ParameterTexture&& other);
    ParameterTexture& operator=(ParameterTexture const& other);
    ParameterTexture& operator=(ParameterTexture&& other);

    void Load(char const* path);
    float Pixel(int ix, int iy) const;
    float Sample(float u, float v) const;
    bool Empty() const;
};
