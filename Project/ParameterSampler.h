#pragma once

#include "SamplerBase.h"
#include "Camera.h"

class ParameterSampler: public SamplerBase
{
private:
    Camera camera;

public:
    ParameterSampler(int width, int height);
    ~ParameterSampler();

    virtual void IteratePixel(int ix, int iy) override;
};
