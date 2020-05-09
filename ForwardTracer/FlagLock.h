#pragma once

#include <atomic>

class FlagLock
{
private:
    std::atomic_flag f;

public:
    FlagLock()
    {
        f.clear(std::memory_order_relaxed);
    }

    ~FlagLock()
    {
    }

    void lock()
    {
        while (!f.test_and_set(std::memory_order_acquire)) {
        }
    }

    void unlock()
    {
        f.clear(std::memory_order_release);
    }
};
