#pragma once
#include <windows.h>
#include <atomic>
#include <thread>

namespace FaceUnlock {

// 120+ FPS (ProMotion / Native Monitor Hz) per-pixel alpha-blended iOS Dynamic Island HUD.
// Driven by QueryPerformanceCounter (sub-microsecond precision) + NT Kernel High-Resolution
// Waitable Timer + sub-pixel GDI+ vector rendering.
// Safe on both WinSta0\Winlogon (LogonUI.exe) and WinSta0\Default.
class FaceIdHud {
public:
    FaceIdHud();
    ~FaceIdHud();

    FaceIdHud(const FaceIdHud&) = delete;
    FaceIdHud& operator=(const FaceIdHud&) = delete;

    // Spawns the 120+ FPS layered topmost HUD thread in Scanning state.
    void StartScanning();

    // Ensures minimum scan animation smoothness, plays Success (green checkmark)
    // or Failed (coral-red iOS horizontal shake), retracts into the top bezel,
    // and joins the UI thread cleanly before returning.
    void FinishWithResult(bool success);

    // Immediate cleanup if aborted.
    void StopImmediate();

private:
    enum class Phase : int {
        Scanning = 0,
        Success  = 1,
        Failed   = 2,
        Retract  = 3,
        Done     = 4
    };

    void ThreadMain();

    std::thread m_thread;
    std::atomic<int> m_phase;
    std::atomic<bool> m_abort;
    LARGE_INTEGER m_startQpc;
    LARGE_INTEGER m_qpcFreq;
};

// Creates a 128x128 32-bpp PARGB HBITMAP depicting the Face ID icon for the
// Windows Lock Screen credential tile. Caller (LogonUI) owns the returned HBITMAP.
HBITMAP CreateFaceIdTileBitmap();

}  // namespace FaceUnlock

extern "C" __declspec(dllexport) void WINAPI PreviewFaceIdHud(BOOL success);
extern "C" __declspec(dllexport) void* WINAPI FaceIdHudStart();
extern "C" __declspec(dllexport) void WINAPI FaceIdHudFinish(void* handle, BOOL success);
