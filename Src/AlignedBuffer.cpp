#include "AlignedBuffer.h"
#include <windows.h>

AlignedBuffer::AlignedBuffer() {
    _size = 0;
    _data = nullptr;
}

AlignedBuffer::AlignedBuffer(size_t size) {
    _size = size;
    if (size != 0) {
        _data = (char*)VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    } else {
        _data = nullptr;
    }
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept {
    _size = other._size;
    _data = other._data;
    other._size = 0;
    other._data = nullptr;
}

AlignedBuffer::~AlignedBuffer() {
    if (_data) {
        VirtualFree(_data, 0, MEM_RELEASE);
    }
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) {
    if (_data) {
        VirtualFree(_data, 0, MEM_RELEASE);
    }
    _size = other._size;
    _data = other._data;
    other._size = 0;
    other._data = nullptr;
    return *this;
}

char* AlignedBuffer::data() {
    return _data;
}

char const* AlignedBuffer::data() const {
    return _data;
}

size_t AlignedBuffer::size() const {
    return _size;
}
