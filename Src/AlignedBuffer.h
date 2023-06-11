#pragma once

#include <stdint.h>

class AlignedBuffer {
private:
    char* _data;
    size_t _size;

public:
    AlignedBuffer();
    AlignedBuffer(size_t size);
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    ~AlignedBuffer();
    AlignedBuffer& operator=(AlignedBuffer&& other);

    char* data();
    size_t size();
};
