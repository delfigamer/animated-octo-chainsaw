#define _CRT_SECURE_NO_WARNINGS

#include "Texture.h"
//#include <png.h>
#include <cmath>
#include <cstdio>
#include <csetjmp>

enum
{
    Nearest,
    Linear,
    Cubic,
    Lanczos2,
    Lanczos3,
    Bad2,
    Bad3,
};

constexpr int InterpolationMethod = Cubic;

static float SrgbTable[256] = { 0 };

void InitializeSrgbTable()
{
    if (SrgbTable[1] != 0) {
        return;
    }
    for (int i = 0; i < 11; ++i) {
        float u = i / 255.0f;
        SrgbTable[i] = 25.0f / 323.0f * u;
    }
    for (int i = 11; i < 256; ++i) {
        float u = i / 255.0f;
        SrgbTable[i] = powf((200.0f * u + 11.0f) / 211.0f, 2.4f);
    }
}

static FDisp CubicInterpolate(FDisp dm1, FDisp d0, FDisp d1, FDisp d2, float w)
{
    float km1 = ((-0.5f * w + 1.0f) * w - 0.5f) * w;
    float k0 = (1.5f * w - 2.5f) * w * w + 1.0f;
    float k1 = ((-1.5f * w + 2.0f) * w + 0.5f) * w;
    float k2 = (0.5f * w - 0.5f) * w * w;
    return dm1 * km1 + d0 * k0 + d1 * k1 + d2 * k2;
}

static float Lanczos2Kernel(float x)
{
    x = 0.25f * x * x;
    if (x > 1) {
        return 0;
    }
    float xd = 2 * x;
    float b5 = -0.3856309f;
    float b4 = 2.7652151f + xd * b5;
    float b3 = -9.933623f + xd * b4 - b5;
    float b2 = 22.65200f + xd * b3 - b4;
    float b1 = -35.98820f + xd * b2 - b3;
    float r = 20.88544f + x * b1 - b2;
    return r;
}

static float Lanczos3Kernel(float x)
{
    x = 0.1111111f * x * x;
    if (x > 1) {
        return 0;
    }
    float xd = 2 * x;
    float b6 = 1.471298f;
    float b5 = -10.97607f + xd * b6;
    float b4 = 41.89659f + xd * b5 - b6;
    float b3 = -105.8643f + xd * b4 - b5;
    float b2 = 195.8197f + xd * b3 - b4;
    float b1 = -278.7240f + xd * b2 - b3;
    float r = 156.3904f + x * b1 - b2;
    return r;
}

static float SmoothStep(float x)
{
    if (InterpolationMethod == Bad3) {
        return ((6 * x - 15) * x + 10) * x * x * x;
    }
    return (3 - 2 * x) * x * x;
}

template<typename T>
auto GenericSample(T const& self, float u, float v) -> decltype(self.Pixel(0,0))
{
    int ix = (int)floorf(u);
    int iy = (int)floorf(v);
    if (InterpolationMethod == Nearest) {
        return self.Pixel(ix, iy);
    }
    float dx = u - ix;
    float dy = v - iy;
    if (InterpolationMethod == Linear || InterpolationMethod == Bad2 || InterpolationMethod == Bad3) {
        FDisp pll = self.Pixel(ix, iy);
        FDisp pdl = self.Pixel(ix + 1, iy);
        FDisp pld = self.Pixel(ix, iy + 1);
        FDisp pdd = self.Pixel(ix + 1, iy + 1);
        if (InterpolationMethod == Bad2 || InterpolationMethod == Bad3) {
            dx = SmoothStep(dx);
            dy = SmoothStep(dy);
        }
        return
            pll * (1 - dx) * (1 - dy)
            + pdl * dx * (1 - dy)
            + pld * (1 - dx) * dy
            + pdd * dx * dy;
    }
    if (InterpolationMethod == Cubic) {
        FDisp dm1 = CubicInterpolate(
            self.Pixel(ix - 1, iy - 1),
            self.Pixel(ix, iy - 1),
            self.Pixel(ix + 1, iy - 1),
            self.Pixel(ix + 2, iy - 1),
            dx);
        FDisp d0 = CubicInterpolate(
            self.Pixel(ix - 1, iy),
            self.Pixel(ix, iy),
            self.Pixel(ix + 1, iy),
            self.Pixel(ix + 2, iy),
            dx);
        FDisp d1 = CubicInterpolate(
            self.Pixel(ix - 1, iy + 1),
            self.Pixel(ix, iy + 1),
            self.Pixel(ix + 1, iy + 1),
            self.Pixel(ix + 2, iy + 1),
            dx);
        FDisp d2 = CubicInterpolate(
            self.Pixel(ix - 1, iy + 2),
            self.Pixel(ix, iy + 2),
            self.Pixel(ix + 1, iy + 2),
            self.Pixel(ix + 2, iy + 2),
            dx);
        return CubicInterpolate(dm1, d0, d1, d2, dy);
    }
    if (InterpolationMethod == Lanczos2) {
        FDisp acc = FDisp{ 0, 0, 0 };
        float den = 0;
        for (int ny = -1; ny < 2; ++ny) {
            for (int nx = -1; nx < 2; ++nx) {
                float weight = Lanczos2Kernel(dx - nx) * Lanczos2Kernel(dy - ny);
                acc += self.Pixel(ix + nx, iy + ny) * weight;
                den += weight;
            }
        }
        return acc / den;
    }
    if (InterpolationMethod == Lanczos3) {
        FDisp acc = FDisp{ 0, 0, 0 };
        float den = 0;
        for (int ny = -2; ny < 3; ++ny) {
            for (int nx = -2; nx < 3; ++nx) {
                float weight = Lanczos3Kernel(dx - nx) * Lanczos3Kernel(dy - ny);
                acc += self.Pixel(ix + nx, iy + ny) * weight;
                den += weight;
            }
        }
        return acc;
    }
}

ColorTexture::ColorTexture()
{
    InitializeSrgbTable();
    width = 0;
    height = 0;
}

ColorTexture::~ColorTexture()
{
}

ColorTexture::ColorTexture(ColorTexture const& other)
{
    *this = other;
}

ColorTexture::ColorTexture(ColorTexture&& other)
{
    *this = std::move(other);
}

ColorTexture& ColorTexture::operator=(ColorTexture const& other)
{
    width = other.width;
    height = other.height;
    pixels = other.pixels;
    return *this;
}

ColorTexture& ColorTexture::operator=(ColorTexture&& other)
{
    width = other.width;
    height = other.height;
    pixels = std::move(other.pixels);
    other.width = 0;
    other.height = 0;
    return *this;
}

//static void texload_error_handler(png_structp png, png_const_charp msg)
//{
//    jmp_buf* buf = (jmp_buf*)png_get_error_ptr(png);
//    longjmp(*buf, 1);
//}
//
//static void texload_warning_handler(png_structp png, png_const_charp msg)
//{
//}

void ColorTexture::Load(char const* path)
{
    //jmp_buf jbuf;
    //FILE* fin = fopen(path, "rb");
    //if (!fin) {
    //    return;
    //}
    //png_structp png = png_create_read_struct(
    //    PNG_LIBPNG_VER_STRING, &jbuf,
    //    &texload_error_handler, &texload_warning_handler);
    //if (png) {
    //    png_infop info = png_create_info_struct(png);
    //    if (info) {
    //        if (!setjmp(jbuf)) {
    //            png_init_io(png, fin);
    //            png_read_info(png, info);
    //            png_uint_32 width;
    //            png_uint_32 height;
    //            int bitdepth;
    //            int colortype;
    //            png_get_IHDR(png, info,
    //                &width, &height,
    //                &bitdepth, &colortype, 0, 0, 0);
    //            double gamma;
    //            if (png_get_gAMA(png, info, &gamma)) {
    //                png_set_gamma(png, 2.2, gamma);
    //            } else {
    //                png_set_gamma(png, 2.2, 0.45455);
    //            }
    //            if (colortype == PNG_COLOR_TYPE_PALETTE) {
    //                png_set_palette_to_rgb(png);
    //            }
    //            if (colortype == PNG_COLOR_TYPE_GRAY && bitdepth < 8) {
    //                png_set_expand_gray_1_2_4_to_8(png);
    //            }
    //            if (png_get_valid(png, info, PNG_INFO_tRNS)) {
    //                png_set_tRNS_to_alpha(png);
    //            } else {
    //                int channels = png_get_channels(png, info);
    //                if (channels == 1 || channels == 3) {
    //                    png_set_add_alpha(png, 255, PNG_FILLER_AFTER);
    //                }
    //            }
    //            if (colortype == PNG_COLOR_TYPE_GRAY ||
    //                colortype == PNG_COLOR_TYPE_GRAY_ALPHA) {
    //                png_set_gray_to_rgb(png);
    //            }
    //            if (bitdepth == 16) {
    //                png_set_scale_16(png);
    //            }
    //            png_set_interlace_handling(png);
    //            png_read_update_info(png, info);
    //            std::vector<uint8_t> pixels(width * height * 4, 0);
    //            std::vector<uint8_t*> rowpointers(height);
    //            for (unsigned y = 0; y < height; ++y) {
    //                rowpointers[y] =
    //                    pixels.data() + y * width * 4;
    //            }
    //            png_read_image(png, (png_bytep*)rowpointers.data());
    //            png_read_end(png, nullptr);
    //            this->width = width;
    //            this->height = height;
    //            this->pixels = std::move(pixels);
    //        }
    //    }
    //    png_destroy_read_struct(
    //        &png,
    //        info ? &info : nullptr,
    //        nullptr);
    //}
    //fclose(fin);
}

FDisp ColorTexture::Pixel(int ix, int iy) const
{
    if (width == 0 || height == 0) {
        return FDisp{ 0, 0, 0 };
    }
    ix %= width;
    if (ix < 0) {
        ix += width;
    }
    iy %= height;
    if (iy < 0) {
        iy += height;
    }
    uint8_t const* p = pixels.data() + iy * width * 4 + ix * 4;
    return FDisp{ SrgbTable[p[0]], SrgbTable[p[1]], SrgbTable[p[2]] };
}

FDisp ColorTexture::Sample(float u, float v) const
{
    return GenericSample(*this, u, v);
}

bool ColorTexture::Empty() const
{
    return pixels.empty();
}
