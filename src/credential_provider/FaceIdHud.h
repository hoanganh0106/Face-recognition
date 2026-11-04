#pragma once
#include <windows.h>
#include <atomic>
#include <mutex>
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

    // Spawns the 120+ FPS layered topmost HUD thread in Scanning state, or switches an
    // island that is still visible (Idle after a failed scan) back to Scanning in place.
    void StartScanning();

    // Ensures minimum scan animation smoothness, then plays Success (green checkmark)
    // or Failed (coral-red iOS horizontal shake).
    // Success, or Failed with stayVisibleOnFail == false: retracts into the top bezel and
    // joins the UI thread before returning.
    // Failed with stayVisibleOnFail == true: returns immediately; the island fades from red
    // back to the scanning accent and stays on screen in Idle until StartScanning() or Hide().
    void FinishWithResult(bool success, bool stayVisibleOnFail = false);

    // Retracts a visible island without blocking the caller.
    void Hide();

    // Immediate cleanup if aborted.
    void StopImmediate();

private:
    enum class Phase : int {
        Scanning = 0,
        Success  = 1,
        Failed   = 2,
        Idle     = 3,
        Retract  = 4
    };

    void ThreadMain();

    std::thread m_thread;
    std::mutex m_mutex;
    std::atomic<int> m_phase;
    std::atomic<bool> m_abort;
    std::atomic<bool> m_running;
    std::atomic<bool> m_stayOnFail;
    LARGE_INTEGER m_startQpc;
    LARGE_INTEGER m_qpcFreq;
};

// Creates a 128x128 32-bpp PARGB HBITMAP depicting the Face ID icon for the
// Windows Lock Screen credential tile. Caller (LogonUI) owns the returned HBITMAP.
HBITMAP CreateFaceIdTileBitmap();

// iOS-style white Face ID glyph on a transparent 72x72 bottom-up HBITMAP with straight alpha.
HBITMAP CreateFaceIdSignInOptionBitmap();

}  // namespace FaceUnlock

extern "C" __declspec(dllexport) void WINAPI PreviewFaceIdHud(BOOL success);
extern "C" __declspec(dllexport) void* WINAPI FaceIdHudStart();
extern "C" __declspec(dllexport) void WINAPI FaceIdHudFinish(void* handle, BOOL success);
