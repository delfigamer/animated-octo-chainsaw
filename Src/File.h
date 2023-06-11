#pragma once

#include <cstdio>
#include <cstdint>
#include <string>
#include <stdexcept>

struct File {
    FILE* handle = nullptr;

    File(std::string const& path, char const* mode);
    ~File();

    template<typename T>
    T read() {
        T var;
        size_t r = fread(&var, sizeof(T), 1, handle);
        if (r == 0) {
            throw std::runtime_error("Read failure");
        }
        return var;
    }

    void read_bytes(size_t count, char* buffer) {
        size_t r = fread(buffer, 1, count, handle);
        if (r != count) {
            throw std::runtime_error("Read failure");
        }
    }
};
