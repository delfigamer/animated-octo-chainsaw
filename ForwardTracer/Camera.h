#pragma once

#include "FLin.h"

struct Camera
{
    FMat mwc;
    FMat mcw;
    float utan;
    float vtan;

    static Camera Targeted(FPoint const& origin, FPoint const& target, float utan, float vtan)
    {
        FDisp forward = norm(target - origin);
        FDisp up = FDisp{ 0, 0, 1 };
        FDisp right = norm(cross(forward, up));
        FDisp down = norm(cross(forward, right));
        Camera c;
        c.mwc = FMat::basis(right, down, forward, origin);
        c.mcw = inverse(c.mwc);
        c.utan = utan;
        c.vtan = vtan;
        return c;
    }
};

