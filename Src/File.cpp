#define _CRT_SECURE_NO_WARNINGS

#include "File.h"

File::File(std::string const& path, char const* mode) {
    handle = fopen(path.c_str(), mode);
    if (!handle) {
        throw std::runtime_error("Failed to open file: " + path);
    }
}

File::~File() {
    if (handle) {
        fclose(handle);
    }
}
