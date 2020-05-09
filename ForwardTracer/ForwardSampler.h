#pragma once

#include "SamplerBase.h"
#include "Camera.h"

class ForwardSampler: public SamplerBase
{
private:
    Camera camera;

public:
    ForwardSampler(int width, int height);
    ~ForwardSampler();

    virtual void IteratePixel(int ix, int iy) override;
};
