#pragma once

#include "header.h"
#include "TripleBuffer.h"
#include <vector>
#include <thread>
#include <atomic>

class RendererThread
{
private:
    HWND hwnd;
    TripleBuffer<std::vector<uint8_t>> bitbuf;
    std::thread rthread;
    std::atomic<bool> rterminate;
    std::atomic<bool> exportrequested;
    std::atomic<bool> pauserequested;

    void ThreadFunc();

public:
    RendererThread(HWND hwnd);
    ~RendererThread();

    void GetWindowSize(int& w, int& h);
    void DrawFrame(HDC dc);
    void Export();
    void SetPause(bool value);
};

