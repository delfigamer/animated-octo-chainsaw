#pragma once

#include "FlagLock.h"
#include <mutex>

template<typename T>
class TripleBuffer
{
private:
    T buf[3];
    unsigned char back;
    unsigned char mid;
    unsigned char fwd;
    bool dirty;
    FlagLock mutex;

public:
    TripleBuffer()
    {
        back = 0;
        mid = 1;
        fwd = 2;
        dirty = false;
    }

    ~TripleBuffer()
    {
    }

    TripleBuffer(TripleBuffer const&) = delete;
    TripleBuffer& operator=(TripleBuffer const&) = delete;

    T& Forward()
    {
        std::unique_lock<FlagLock> ul(mutex);
        if (dirty) {
            std::swap(mid, fwd);
            dirty = false;
        }
        return buf[fwd];
    }

    T& Back()
    {
        return buf[back];
    }

    void Publish()
    {
        std::unique_lock<FlagLock> ul(mutex);
        std::swap(back, mid);
        dirty = true;
    }
};
