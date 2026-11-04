#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>

#include "FaceIdHud.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <new>

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace FaceUnlock {

namespace {

constexpr float kPi = 3.14159265358979323846f;

inline float Clamp01(float v) noexcept {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline float EaseOutCubic(float t) noexcept {
    t = Clamp01(t);
    float inv = 1.0f - t;
    return 1.0f - inv * inv * inv;
}

inline float EaseInCubic(float t) noexcept {
    t = Clamp01(t);
    return t * t * t;
}

// Continuous damped spring matching Apple ProMotion 120Hz Dynamic Island feel
inline float SpringExpand(float t) noexcept {
    t = Clamp01(t);
    if (t >= 1.0f) return 1.0f;
    return 1.0f - std::exp(-8.8f * t) * std::cos(10.8f * t);
}

inline double ElapsedSecondsQpc(const LARGE_INTEGER& start,
                                const LARGE_INTEGER& now,
                                const LARGE_INTEGER& freq) noexcept {
    return static_cast<double>(now.QuadPart - start.QuadPart) / static_cast<double>(freq.QuadPart);
}

// Fast vectorizable premultiplied alpha conversion for UpdateLayeredWindow (AC_SRC_ALPHA)
void PremultiplyAlphaBuffer(uint32_t* __restrict pixels, int count) noexcept {
    for (int i = 0; i < count; ++i) {
        uint32_t px = pixels[i];
        uint32_t a = px >> 24;
        if (a == 0u) {
            pixels[i] = 0u;
        } else if (a < 255u) {
            uint32_t r = (((px >> 16) & 0xFFu) * a + 127u) / 255u;
            uint32_t g = (((px >> 8)  & 0xFFu) * a + 127u) / 255u;
            uint32_t b = (((px)       & 0xFFu) * a + 127u) / 255u;
            pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

void AddRoundedRect(Gdiplus::GraphicsPath& path, float x, float y, float w, float h, float r) {
    float maxR = (std::min)(w, h) * 0.5f;
    if (r > maxR) r = maxR;
    float d = r * 2.0f;
    if (d <= 0.5f) {
        path.AddRectangle(Gdiplus::RectF(x, y, w, h));
        return;
    }
    path.AddArc(x, y, d, d, 180.0f, 90.0f);
    path.AddArc(x + w - d, y, d, d, 270.0f, 90.0f);
    path.AddArc(x + w - d, y + h - d, d, d, 0.0f, 90.0f);
    path.AddArc(x, y + h - d, d, d, 90.0f, 90.0f);
    path.CloseFigure();
}

void DrawFaceIdGlyph(Gdiplus::Graphics& g,
                     float cx,
                     float cy,
                     float size,
                     float timeSec,
                     const Gdiplus::Color& primaryCol,
                     float alphaScale,
                     bool drawScanBeam,
                     bool drawOuterBrackets = true,
                     float strokeScale = 1.0f) {
    auto withAlpha = [&](const Gdiplus::Color& c, float mult) -> Gdiplus::Color {
        int a = static_cast<int>(std::round(c.GetA() * Clamp01(alphaScale * mult)));
        return Gdiplus::Color(static_cast<BYTE>((std::clamp)(a, 0, 255)), c.GetR(), c.GetG(), c.GetB());
    };

    float breathe = 1.0f + 0.045f * std::sin(timeSec * 6.8f);
    float half = (size * 0.5f) * breathe;
    float cornerR = size * 0.22f;
    float armLen = size * 0.28f;
    float stroke = (std::max)(2.0f, size * 0.082f * strokeScale);

    float left = cx - half;
    float top = cy - half;
    float right = cx + half;
    float bottom = cy + half;
    float d = cornerR * 2.0f;

    if (drawOuterBrackets) {
        Gdiplus::Pen bracketPen(withAlpha(primaryCol, 1.0f), stroke);
        bracketPen.SetStartCap(Gdiplus::LineCapRound);
        bracketPen.SetEndCap(Gdiplus::LineCapRound);
        bracketPen.SetLineJoin(Gdiplus::LineJoinRound);

        {
            Gdiplus::GraphicsPath p;
            p.AddLine(left, top + armLen, left, top + cornerR);
            p.AddArc(left, top, d, d, 180.0f, 90.0f);
            p.AddLine(left + cornerR, top, left + armLen, top);
            g.DrawPath(&bracketPen, &p);
        }
        {
            Gdiplus::GraphicsPath p;
            p.AddLine(right - armLen, top, right - cornerR, top);
            p.AddArc(right - d, top, d, d, 270.0f, 90.0f);
            p.AddLine(right, top + cornerR, right, top + armLen);
            g.DrawPath(&bracketPen, &p);
        }
        {
            Gdiplus::GraphicsPath p;
            p.AddLine(right, bottom - armLen, right, bottom - cornerR);
            p.AddArc(right - d, bottom - d, d, d, 0.0f, 90.0f);
            p.AddLine(right - cornerR, bottom, right - armLen, bottom);
            g.DrawPath(&bracketPen, &p);
        }
        {
            Gdiplus::GraphicsPath p;
            p.AddLine(left + armLen, bottom, left + cornerR, bottom);
            p.AddArc(left, bottom - d, d, d, 90.0f, 90.0f);
            p.AddLine(left, bottom - cornerR, left, bottom - armLen);
            g.DrawPath(&bracketPen, &p);
        }
    }

    float px = std::sin(timeSec * 4.2f) * (size * 0.048f);
    float py = std::cos(timeSec * 3.1f) * (size * 0.032f);
    float fcx = cx + px;
    float fcy = cy + py;

    Gdiplus::Pen featurePen(withAlpha(primaryCol, 0.94f), stroke * 0.88f);
    featurePen.SetStartCap(Gdiplus::LineCapRound);
    featurePen.SetEndCap(Gdiplus::LineCapRound);
    featurePen.SetLineJoin(Gdiplus::LineJoinRound);

    float eyeX = size * 0.16f;
    float eyeTop = fcy - size * 0.16f;
    float eyeBot = fcy - size * 0.03f;
    g.DrawLine(&featurePen, fcx - eyeX, eyeTop, fcx - eyeX, eyeBot);
    g.DrawLine(&featurePen, fcx + eyeX, eyeTop, fcx + eyeX, eyeBot);

    {
        Gdiplus::GraphicsPath nose;
        float nw = size * 0.06f;
        nose.AddLine(fcx + nw * 0.3f, fcy - size * 0.10f, fcx + nw * 0.3f, fcy + size * 0.05f);
        nose.AddArc(fcx - nw, fcy + size * 0.01f, nw * 1.3f, size * 0.08f, 0.0f, 115.0f);
        g.DrawPath(&featurePen, &nose);
    }

    {
        float sw = size * 0.32f;
        float sh = size * 0.18f;
        g.DrawArc(&featurePen, fcx - sw * 0.5f, fcy + size * 0.06f, sw, sh, 20.0f, 140.0f);
    }

    if (drawScanBeam) {
        float sweepPhase = 0.5f + 0.5f * std::sin(timeSec * 5.4f);
        float beamY = top + (bottom - top) * sweepPhase;
        float beamH = size * 0.25f;

        Gdiplus::GraphicsPath clipPath;
        AddRoundedRect(clipPath, left + stroke * 0.5f, top + stroke * 0.5f,
                       (right - left) - stroke, (bottom - top) - stroke, cornerR * 0.8f);
        g.SetClip(&clipPath);

        Gdiplus::LinearGradientBrush beamBrush(
            Gdiplus::PointF(cx, beamY - beamH * 0.5f),
            Gdiplus::PointF(cx, beamY + beamH * 0.5f),
            Gdiplus::Color(0, 50, 215, 255),
            Gdiplus::Color(0, 50, 215, 255));
        Gdiplus::Color colors[3] = {
            Gdiplus::Color(0, 50, 215, 255),
            withAlpha(Gdiplus::Color(115, 90, 230, 255), 1.0f),
            Gdiplus::Color(0, 50, 215, 255)
        };
        Gdiplus::REAL positions[3] = { 0.0f, 0.5f, 1.0f };
        beamBrush.SetInterpolationColors(colors, positions, 3);
        g.FillRectangle(&beamBrush, left, beamY - beamH * 0.5f, right - left, beamH);

        Gdiplus::Pen linePen(withAlpha(Gdiplus::Color(205, 145, 245, 255), 1.0f),
                             (std::max)(1.2f, stroke * 0.38f));
        g.DrawLine(&linePen, left + stroke, beamY, right - stroke, beamY);
        g.ResetClip();
    }
}

void DrawSuccessCheckGlyph(Gdiplus::Graphics& g,
                           float cx,
                           float cy,
                           float size,
                           float progress,
                           float alphaScale) {
    float p = EaseOutCubic(progress);
    BYTE aRing = static_cast<BYTE>(std::round(255.0f * Clamp01(alphaScale)));
    BYTE aFill = static_cast<BYTE>(std::round(46.0f * Clamp01(alphaScale)));
    BYTE aGlow = static_cast<BYTE>(std::round(34.0f * Clamp01(alphaScale) * p));

    float r = size * 0.48f * (0.86f + 0.14f * p);

    Gdiplus::SolidBrush glowBrush(Gdiplus::Color(aGlow, 48, 209, 88));
    float gr = r * 1.28f;
    g.FillEllipse(&glowBrush, cx - gr, cy - gr, gr * 2.0f, gr * 2.0f);

    Gdiplus::SolidBrush fillBrush(Gdiplus::Color(aFill, 48, 209, 88));
    g.FillEllipse(&fillBrush, cx - r, cy - r, r * 2.0f, r * 2.0f);

    float ringStroke = (std::max)(2.0f, size * 0.075f);
    Gdiplus::Pen ringPen(Gdiplus::Color(aRing, 48, 209, 88), ringStroke);
    g.DrawEllipse(&ringPen, cx - r, cy - r, r * 2.0f, r * 2.0f);

    Gdiplus::PointF p0(cx - r * 0.38f, cy + r * 0.02f);
    Gdiplus::PointF p1(cx - r * 0.08f, cy + r * 0.32f);
    Gdiplus::PointF p2(cx + r * 0.42f, cy - r * 0.26f);

    Gdiplus::Pen checkPen(Gdiplus::Color(aRing, 70, 235, 110), (std::max)(2.4f, size * 0.095f));
    checkPen.SetStartCap(Gdiplus::LineCapRound);
    checkPen.SetEndCap(Gdiplus::LineCapRound);
    checkPen.SetLineJoin(Gdiplus::LineJoinRound);

    if (p > 0.02f) {
        Gdiplus::GraphicsPath cp;
        const float seg1Frac = 0.38f;
        if (p <= seg1Frac) {
            float k = p / seg1Frac;
            Gdiplus::PointF mid(p0.X + (p1.X - p0.X) * k, p0.Y + (p1.Y - p0.Y) * k);
            cp.AddLine(p0, mid);
        } else {
            float k = (p - seg1Frac) / (1.0f - seg1Frac);
            Gdiplus::PointF end(p1.X + (p2.X - p1.X) * k, p1.Y + (p2.Y - p1.Y) * k);
            cp.AddLine(p0, p1);
            cp.AddLine(p1, end);
        }
        g.DrawPath(&checkPen, &cp);
    }
}

LRESULT CALLBACK HudWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_ERASEBKGND) return 1;
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

}  // namespace

FaceIdHud::FaceIdHud()
    : m_phase(static_cast<int>(Phase::Scanning)),
      m_abort(false),
      m_running(false),
      m_stayOnFail(false),
      m_startQpc{},
      m_qpcFreq{} {
    QueryPerformanceFrequency(&m_qpcFreq);
}

FaceIdHud::~FaceIdHud() {
    StopImmediate();
}

void FaceIdHud::StartScanning() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        QueryPerformanceCounter(&m_startQpc);
        if (m_running.load(std::memory_order_acquire)) {
            m_phase.store(static_cast<int>(Phase::Scanning), std::memory_order_release);
            return;
        }
    }

    StopImmediate();
    m_abort.store(false, std::memory_order_release);
    m_phase.store(static_cast<int>(Phase::Scanning), std::memory_order_release);
    m_running.store(true, std::memory_order_release);
    try {
        m_thread = std::thread(&FaceIdHud::ThreadMain, this);
    } catch (...) {
        m_running.store(false, std::memory_order_release);
    }
}

void FaceIdHud::FinishWithResult(bool success, bool stayVisibleOnFail) {
    if (!m_running.load(std::memory_order_acquire)) return;

    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    double elapsedMs = ElapsedSecondsQpc(m_startQpc, now, m_qpcFreq) * 1000.0;
    if (elapsedMs < 320.0) {
        Sleep(static_cast<DWORD>(320.0 - elapsedMs));
    }

    const bool keepVisible = !success && stayVisibleOnFail;
    m_stayOnFail.store(keepVisible, std::memory_order_release);
    m_phase.store(static_cast<int>(success ? Phase::Success : Phase::Failed),
                  std::memory_order_release);

    if (keepVisible) return;

    // LogonUI blocks on this call, so never wait on the animation for more than a few seconds.
    const ULONGLONG deadline = GetTickCount64() + 3000ULL;
    while (m_running.load(std::memory_order_acquire) && GetTickCount64() < deadline) {
        Sleep(5);
    }
    StopImmediate();
}

void FaceIdHud::Hide() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_running.load(std::memory_order_acquire)) {
        m_phase.store(static_cast<int>(Phase::Retract), std::memory_order_release);
    }
}

void FaceIdHud::StopImmediate() {
    m_abort.store(true, std::memory_order_release);
    if (m_thread.joinable()) {
        m_thread.join();
    }
    m_running.store(false, std::memory_order_release);
}

void FaceIdHud::ThreadMain() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken = 0;
    if (Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr) != Gdiplus::Ok) {
        m_running.store(false, std::memory_order_release);
        return;
    }

    POINT ptZero{ 0, 0 };
    HMONITOR hMon = MonitorFromPoint(ptZero, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    int monX = 0, monY = 0;
    int monW = GetSystemMetrics(SM_CXSCREEN);
    int targetFps = 120;

    if (GetMonitorInfoW(hMon, &mi)) {
        monX = mi.rcMonitor.left;
        monY = mi.rcMonitor.top;
        monW = mi.rcMonitor.right - mi.rcMonitor.left;

        DEVMODEW dm{};
        dm.dmSize = sizeof(dm);
        if (EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm)) {
            if (dm.dmDisplayFrequency >= 120 && dm.dmDisplayFrequency <= 240) {
                targetFps = static_cast<int>(dm.dmDisplayFrequency);
            }
        }
    }

    float scale = std::clamp(monW / 1920.0f, 1.0f, 1.85f);

    const int canvasW = static_cast<int>(std::round(410.0f * scale));
    const int canvasH = static_cast<int>(std::round(136.0f * scale));
    const int wndX = monX + (monW - canvasW) / 2;
    const int wndY = monY + static_cast<int>(std::round(8.0f * scale));

    const wchar_t* clsName = L"FaceUnlockDynamicIslandProMotionWnd";
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = HudWndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = clsName;
    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT,
        clsName,
        L"",
        WS_POPUP,
        wndX, wndY, canvasW, canvasH,
        nullptr, nullptr, wc.hInstance, nullptr);

    if (!hwnd) {
        Gdiplus::GdiplusShutdown(gdiplusToken);
        m_running.store(false, std::memory_order_release);
        return;
    }

    ShowWindow(hwnd, SW_SHOWNOACTIVATE);
    SetWindowPos(hwnd, HWND_TOPMOST, wndX, wndY, canvasW, canvasH,
                 SWP_NOACTIVATE | SWP_SHOWWINDOW);

    HDC screenDc = GetDC(nullptr);
    HDC memDc = CreateCompatibleDC(screenDc);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = canvasW;
    bmi.bmiHeader.biHeight = -canvasH;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* dibBits = nullptr;
    HBITMAP hDib = CreateDIBSection(screenDc, &bmi, DIB_RGB_COLORS, &dibBits, nullptr, 0);
    HGDIOBJ oldBmp = hDib ? SelectObject(memDc, hDib) : nullptr;

    HANDLE hHiResTimer = CreateWaitableTimerExW(
        nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!hHiResTimer) {
        hHiResTimer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
    }

    if (hDib && dibBits) {
        Gdiplus::Bitmap gdiBmp(canvasW, canvasH, canvasW * 4, PixelFormat32bppARGB,
                               static_cast<BYTE*>(dibBits));
        Gdiplus::FontFamily fontFamily(L"Segoe UI");
        Gdiplus::Font titleFont(&fontFamily, 14.5f * scale, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
        Gdiplus::Font subFont(&fontFamily, 11.8f * scale, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);

        LARGE_INTEGER qpcStart{};
        QueryPerformanceCounter(&qpcStart);
        LARGE_INTEGER phaseQpc = qpcStart;
        Phase lastRequested = static_cast<Phase>(m_phase.load(std::memory_order_acquire));
        Phase activePhase = lastRequested;
        // Phase whose look is kept while the island retracts.
        Phase retractLook = activePhase;
        bool idleFrameSettled = false;
        LONGLONG lastTopmostQpc = qpcStart.QuadPart;

        const LONGLONG frameStepQpc = (std::max)(1LL, m_qpcFreq.QuadPart / targetFps);
        LONGLONG nextDeadlineQpc = qpcStart.QuadPart;

        constexpr float kFailedSec  = 0.56f;
        constexpr float kIdleFadeSec = 0.45f;
        auto failFlashPulse = [](float t) {
            return std::exp(-3.4f * t) * (0.55f + 0.45f * std::abs(std::sin(t * 17.0f)));
        };

        while (!m_abort.load(std::memory_order_acquire)) {
            MSG msg{};
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }

            LARGE_INTEGER nowQpc{};
            QueryPerformanceCounter(&nowQpc);

            Phase requested = static_cast<Phase>(m_phase.load(std::memory_order_acquire));
            if (requested != lastRequested) {
                lastRequested = requested;
                if (requested == Phase::Retract && activePhase != Phase::Retract) {
                    retractLook = activePhase;
                }
                activePhase = requested;
                phaseQpc = nowQpc;
                idleFrameSettled = false;
            }

            float totalSec = static_cast<float>(ElapsedSecondsQpc(qpcStart, nowQpc, m_qpcFreq));
            float phaseSec = static_cast<float>(ElapsedSecondsQpc(phaseQpc, nowQpc, m_qpcFreq));

            if (activePhase == Phase::Success && phaseSec >= 0.42f) {
                retractLook = Phase::Success;
                activePhase = Phase::Retract;
                phaseQpc = nowQpc;
                phaseSec = 0.0f;
            } else if (activePhase == Phase::Failed && phaseSec >= kFailedSec) {
                if (m_stayOnFail.load(std::memory_order_acquire)) {
                    activePhase = Phase::Idle;
                } else {
                    retractLook = Phase::Failed;
                    activePhase = Phase::Retract;
                }
                phaseQpc = nowQpc;
                phaseSec = 0.0f;
            } else if (activePhase == Phase::Retract && phaseSec >= 0.16f) {
                // Exit unless a new phase was requested while retracting (Success/Failed retract
                // on their own, so the requested phase is not necessarily Retract here).
                std::lock_guard<std::mutex> lock(m_mutex);
                if (static_cast<Phase>(m_phase.load(std::memory_order_acquire)) == lastRequested) {
                    m_running.store(false, std::memory_order_release);
                    break;
                }
                continue;
            }

            if (activePhase == Phase::Idle && idleFrameSettled) {
                if (nowQpc.QuadPart - lastTopmostQpc > m_qpcFreq.QuadPart) {
                    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
                    lastTopmostQpc = nowQpc.QuadPart;
                }
                Sleep(15);
                nextDeadlineQpc = nowQpc.QuadPart;
                continue;
            }
            if (activePhase == Phase::Idle && phaseSec >= kIdleFadeSec + 0.05f) {
                idleFrameSettled = true;
            }

            const Phase look = (activePhase == Phase::Retract) ? retractLook : activePhase;
            // 1 = fully red (end of the failure flash), 0 = back to the scanning accent.
            const float idleRedMix = (activePhase == Phase::Idle)
                                         ? 1.0f - EaseOutCubic(phaseSec / kIdleFadeSec)
                                         : 0.0f;

            const float targetW = 308.0f * scale;
            const float targetH = 84.0f * scale;
            const float closedW = 68.0f * scale;
            const float closedH = 26.0f * scale;

            float morph = 1.0f;
            float masterAlpha = 1.0f;
            float shakeX = 0.0f;
            float liftY = 0.0f;
            float redFlashPulse = 0.0f;

            if (activePhase == Phase::Scanning) {
                morph = SpringExpand(totalSec / 0.35f);
                masterAlpha = Clamp01(totalSec / 0.11f);
            } else if (activePhase == Phase::Success) {
                morph = 1.0f + 0.026f * std::sin(Clamp01(phaseSec / 0.26f) * kPi);
                masterAlpha = 1.0f;
            } else if (activePhase == Phase::Failed) {
                // Pure iOS Dynamic Island crimson flash + horizontal damped shake (no modal swap!)
                morph = 1.0f;
                masterAlpha = 1.0f;
                shakeX = (14.5f * scale) * std::exp(-6.4f * phaseSec) * std::sin(38.0f * phaseSec);
                redFlashPulse = failFlashPulse(phaseSec);
            } else if (activePhase == Phase::Idle) {
                morph = 1.0f;
                masterAlpha = 1.0f;
                redFlashPulse = failFlashPulse(kFailedSec) * idleRedMix;
            } else if (activePhase == Phase::Retract) {
                float rt = EaseInCubic(phaseSec / 0.16f);
                morph = 1.0f - rt;
                masterAlpha = 1.0f - rt;
                liftY = -14.0f * scale * rt;
            }

            float pillW = closedW + (targetW - closedW) * morph;
            float pillH = closedH + (targetH - closedH) * morph;
            float pillX = (canvasW - pillW) * 0.5f + shakeX;
            float pillY = 16.0f * scale + liftY;
            float pillR = pillH * 0.5f;

            std::memset(dibBits, 0, static_cast<size_t>(canvasW) * static_cast<size_t>(canvasH) * 4u);

            {
                Gdiplus::Graphics g(&gdiBmp);
                g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
                g.SetPixelOffsetMode(Gdiplus::PixelOffsetModeHalf);
                g.SetTextRenderingHint(Gdiplus::TextRenderingHintAntiAlias);

                const bool isFailStyle = (look == Phase::Failed);
                const bool isSuccessStyle = (look == Phase::Success);
                const bool isIdleStyle = (look == Phase::Idle);

                auto lerpByte = [](int from, int to, float t) {
                    return static_cast<BYTE>(std::round(from + (to - from) * Clamp01(t)));
                };

                Gdiplus::Color accentCol(255, 45, 212, 255);
                if (isSuccessStyle) {
                    accentCol = Gdiplus::Color(255, 48, 209, 88);
                } else if (isFailStyle) {
                    accentCol = Gdiplus::Color(255, 255, 59, 48);
                } else if (isIdleStyle) {
                    accentCol = Gdiplus::Color(255, lerpByte(45, 255, idleRedMix),
                                               lerpByte(212, 59, idleRedMix),
                                               lerpByte(255, 48, idleRedMix));
                }

                // Ambient halo glow (boosted crimson pulse during Failed flash)
                float haloBoost = (isFailStyle || isIdleStyle) ? (1.0f + 1.35f * redFlashPulse) : 1.0f;
                for (int layer = 3; layer >= 1; --layer) {
                    float pad = layer * 4.2f * scale;
                    int rawA = static_cast<int>(std::round((18.0f * haloBoost / layer) * masterAlpha));
                    BYTE glowA = static_cast<BYTE>((std::clamp)(rawA, 0, 255));
                    Gdiplus::GraphicsPath glowPath;
                    AddRoundedRect(glowPath, pillX - pad, pillY - pad * 0.6f + 2.0f * scale,
                                   pillW + pad * 2.0f, pillH + pad * 1.6f, pillR + pad);
                    Gdiplus::SolidBrush glowBrush(
                        Gdiplus::Color(glowA, accentCol.GetR() / 2, accentCol.GetG() / 2, accentCol.GetB() / 2));
                    g.FillPath(&glowBrush, &glowPath);
                }

                // Pill body (deep obsidian + crimson inner flash tint when scan fails)
                Gdiplus::GraphicsPath bodyPath;
                AddRoundedRect(bodyPath, pillX, pillY, pillW, pillH, pillR);
                BYTE bodyA = static_cast<BYTE>(std::round(244.0f * masterAlpha));
                BYTE bodyR = static_cast<BYTE>(std::round(11.0f + 56.0f * redFlashPulse));
                BYTE bodyG = static_cast<BYTE>(std::round(13.0f + 4.0f  * redFlashPulse));
                BYTE bodyB = static_cast<BYTE>(std::round(19.0f + 4.0f  * redFlashPulse));
                Gdiplus::SolidBrush bodyBrush(Gdiplus::Color(bodyA, bodyR, bodyG, bodyB));
                g.FillPath(&bodyBrush, &bodyPath);

                // Rim highlight
                int topAlphaRaw = static_cast<int>(std::round((78.0f + 115.0f * redFlashPulse) * masterAlpha));
                int botAlphaRaw = static_cast<int>(std::round((24.0f + 70.0f  * redFlashPulse) * masterAlpha));
                BYTE rimTopA = static_cast<BYTE>((std::clamp)(topAlphaRaw, 0, 255));
                BYTE rimBotA = static_cast<BYTE>((std::clamp)(botAlphaRaw, 0, 255));
                Gdiplus::LinearGradientBrush rimBrush(
                    Gdiplus::PointF(pillX, pillY),
                    Gdiplus::PointF(pillX, pillY + pillH),
                    Gdiplus::Color(rimTopA,
                                   static_cast<BYTE>((255 + accentCol.GetR()) / 2),
                                   static_cast<BYTE>((255 + accentCol.GetG()) / 2),
                                   static_cast<BYTE>((255 + accentCol.GetB()) / 2)),
                    Gdiplus::Color(rimBotA, accentCol.GetR(), accentCol.GetG(), accentCol.GetB()));
                Gdiplus::Pen rimPen(&rimBrush, (std::max)(1.2f, (1.35f + 0.55f * redFlashPulse) * scale));
                g.DrawPath(&rimPen, &bodyPath);

                float contentAlpha = Clamp01((morph - 0.35f) / 0.55f) * masterAlpha;
                if (contentAlpha > 0.02f) {
                    g.SetClip(&bodyPath);

                    float iconSize = 44.0f * scale;
                    float iconCx = pillX + 44.0f * scale;
                    float iconCy = pillY + pillH * 0.5f;

                    if (isSuccessStyle) {
                        float cp = (activePhase == Phase::Success) ? (phaseSec / 0.25f) : 1.0f;
                        DrawSuccessCheckGlyph(g, iconCx, iconCy, iconSize, cp, contentAlpha);
                    } else {
                        // Keep the sleek Face ID glyph in BOTH Scanning & Red-Flash Failure (no ugly X icon!)
                        const bool isStatic = isFailStyle || isIdleStyle;
                        DrawFaceIdGlyph(g, iconCx, iconCy, iconSize,
                                        isStatic ? 0.0f : totalSec,
                                        accentCol, contentAlpha,
                                        !isStatic, true);
                    }

                    const wchar_t* titleTxt = L"Face ID";
                    const wchar_t* subTxt   = L"Đang nhận diện khuôn mặt...";
                    Gdiplus::Color subCol(static_cast<BYTE>(std::round(185.0f * contentAlpha)), 156, 168, 188);

                    if (isSuccessStyle) {
                        titleTxt = L"Đã mở khóa";
                        subTxt   = L"Xác thực khuôn mặt thành công";
                        subCol   = Gdiplus::Color(static_cast<BYTE>(std::round(235.0f * contentAlpha)), 65, 222, 105);
                    } else if (isFailStyle) {
                        subTxt   = L"Không nhận ra khuôn mặt";
                        subCol   = Gdiplus::Color(static_cast<BYTE>(std::round(235.0f * contentAlpha)), 255, 115, 108);
                    } else if (isIdleStyle) {
                        subTxt   = L"Chạm hoặc nhấn Space để quét lại";
                        subCol   = Gdiplus::Color(
                            static_cast<BYTE>(std::round((185.0f + 50.0f * idleRedMix) * contentAlpha)),
                            lerpByte(156, 255, idleRedMix),
                            lerpByte(168, 115, idleRedMix),
                            lerpByte(188, 108, idleRedMix));
                    }

                    float textLeft = pillX + 78.0f * scale;
                    float titleY   = pillY + 20.0f * scale;
                    float subY     = pillY + 44.0f * scale;

                    Gdiplus::SolidBrush titleBrush(
                        Gdiplus::Color(static_cast<BYTE>(std::round(250.0f * contentAlpha)), 248, 250, 255));
                    Gdiplus::SolidBrush subBrush(subCol);

                    g.DrawString(titleTxt, -1, &titleFont, Gdiplus::PointF(textLeft, titleY), &titleBrush);
                    g.DrawString(subTxt, -1, &subFont, Gdiplus::PointF(textLeft, subY), &subBrush);

                    if (look == Phase::Scanning) {
                        float dotsBaseX = pillX + pillW - 36.0f * scale;
                        float dotsCy = pillY + 30.0f * scale;
                        for (int d = 0; d < 3; ++d) {
                            float wave = 0.5f + 0.5f * std::sin(totalSec * 8.2f - d * 1.1f);
                            float dr = (2.2f + 1.3f * wave) * scale;
                            BYTE da = static_cast<BYTE>(std::round((90.0f + 155.0f * wave) * contentAlpha));
                            Gdiplus::SolidBrush dotBrush(Gdiplus::Color(da, 50, 215, 255));
                            float dx = dotsBaseX + d * 8.5f * scale;
                            g.FillEllipse(&dotBrush, dx - dr, dotsCy - dr, dr * 2.0f, dr * 2.0f);
                        }
                    }

                    g.ResetClip();
                }
            }

            PremultiplyAlphaBuffer(static_cast<uint32_t*>(dibBits), canvasW * canvasH);

            POINT ptPos{ wndX, wndY };
            SIZE szWnd{ canvasW, canvasH };
            POINT ptSrc{ 0, 0 };
            BLENDFUNCTION blend{};
            blend.BlendOp = AC_SRC_OVER;
            blend.BlendFlags = 0;
            blend.SourceConstantAlpha = 255;
            blend.AlphaFormat = AC_SRC_ALPHA;

            UpdateLayeredWindow(hwnd, screenDc, &ptPos, &szWnd, memDc, &ptSrc, 0, &blend, ULW_ALPHA);

            nextDeadlineQpc += frameStepQpc;
            LARGE_INTEGER afterRenderQpc{};
            QueryPerformanceCounter(&afterRenderQpc);
            LONGLONG remainingTicks = nextDeadlineQpc - afterRenderQpc.QuadPart;
            const LONGLONG spinMarginTicks = m_qpcFreq.QuadPart / 3000LL;  // ~0.33 ms

            if (remainingTicks > 0) {
                if (hHiResTimer && remainingTicks > spinMarginTicks) {
                    LONGLONG sleepTicks = remainingTicks - spinMarginTicks;
                    LONGLONG wait100ns = -(sleepTicks * 10000000LL) / m_qpcFreq.QuadPart;
                    if (wait100ns < -1000LL) {
                        LARGE_INTEGER dueTime{};
                        dueTime.QuadPart = wait100ns;
                        if (SetWaitableTimer(hHiResTimer, &dueTime, 0, nullptr, nullptr, FALSE)) {
                            WaitForSingleObject(hHiResTimer, 15);
                        }
                    }
                } else if (!hHiResTimer && remainingTicks > m_qpcFreq.QuadPart / 500LL) {
                    Sleep(1);
                }

                for (;;) {
                    LARGE_INTEGER spinQpc{};
                    QueryPerformanceCounter(&spinQpc);
                    if (spinQpc.QuadPart >= nextDeadlineQpc) break;
                    _mm_pause();
                }
            } else if (remainingTicks < -frameStepQpc * 2) {
                nextDeadlineQpc = afterRenderQpc.QuadPart;
            }
        }
    }

    if (hHiResTimer) CloseHandle(hHiResTimer);
    if (oldBmp) SelectObject(memDc, oldBmp);
    if (hDib) DeleteObject(hDib);
    if (memDc) DeleteDC(memDc);
    if (screenDc) ReleaseDC(nullptr, screenDc);
    DestroyWindow(hwnd);

    Gdiplus::GdiplusShutdown(gdiplusToken);
}

// 100% Circular frosted-glass Lock Screen avatar (zero square corners!)
HBITMAP CreateFaceIdTileBitmap() {
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR token = 0;
    if (Gdiplus::GdiplusStartup(&token, &gdiplusStartupInput, nullptr) != Gdiplus::Ok) {
        return nullptr;
    }

    const int sz = 128;
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = sz;
    bmi.bmiHeader.biHeight = -sz;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    HDC screenDc = GetDC(nullptr);
    void* bits = nullptr;
    HBITMAP hBmp = CreateDIBSection(screenDc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    ReleaseDC(nullptr, screenDc);

    if (hBmp && bits) {
        std::memset(bits, 0, sz * sz * 4);
        {
            Gdiplus::Bitmap bmp(sz, sz, sz * 4, PixelFormat32bppARGB, static_cast<BYTE*>(bits));
            Gdiplus::Graphics g(&bmp);
            g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
            g.SetPixelOffsetMode(Gdiplus::PixelOffsetModeHalf);

            // Pure circular glass disc matching Windows 11 circular user profile geometry
            Gdiplus::LinearGradientBrush discBrush(
                Gdiplus::PointF(12.0f, 12.0f),
                Gdiplus::PointF(116.0f, 116.0f),
                Gdiplus::Color(220, 18, 28, 48),
                Gdiplus::Color(235, 9, 13, 22));
            g.FillEllipse(&discBrush, 8.0f, 8.0f, 112.0f, 112.0f);

            // Subtle dual circular glowing rings (no square corners!)
            Gdiplus::Pen outerRing(Gdiplus::Color(155, 50, 212, 255), 2.4f);
            g.DrawEllipse(&outerRing, 8.0f, 8.0f, 112.0f, 112.0f);

            Gdiplus::Pen innerArcPen(Gdiplus::Color(90, 90, 225, 255), 1.8f);
            innerArcPen.SetStartCap(Gdiplus::LineCapRound);
            innerArcPen.SetEndCap(Gdiplus::LineCapRound);
            g.DrawArc(&innerArcPen, 18.0f, 18.0f, 92.0f, 92.0f, -45.0f, 90.0f);
            g.DrawArc(&innerArcPen, 18.0f, 18.0f, 92.0f, 92.0f, 135.0f, 90.0f);

            // Draw only the clean curved facial features inside the circle (drawOuterBrackets = false)
            DrawFaceIdGlyph(g, 64.0f, 65.0f, 68.0f, 0.0f,
                            Gdiplus::Color(255, 70, 222, 255), 1.0f, false, false);
        }
        PremultiplyAlphaBuffer(static_cast<uint32_t*>(bits), sz * sz);
    }

    Gdiplus::GdiplusShutdown(token);
    return hBmp;
}

namespace {

// Windows' own Sign-in options bitmaps (PIN, password, Hello) are 72x72 on this gray.
constexpr int kSignInOptionSize = 72;
void DrawSignInOptionGlyph(Gdiplus::Graphics& g) {
    g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
    g.SetPixelOffsetMode(Gdiplus::PixelOffsetModeHalf);
    DrawFaceIdGlyph(g, 36.0f, 36.0f, 39.7f, 0.0f,
                    Gdiplus::Color(255, 255, 255, 255), 1.0f, false, true, 1.29f);
}

}  // namespace

HBITMAP CreateFaceIdSignInOptionBitmap() {
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR token = 0;
    if (Gdiplus::GdiplusStartup(&token, &gdiplusStartupInput, nullptr) != Gdiplus::Ok) {
        return nullptr;
    }

    const int sz = kSignInOptionSize;
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = sz;
    bmi.bmiHeader.biHeight = sz;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    HDC screenDc = GetDC(nullptr);
    void* bits = nullptr;
    HBITMAP hBmp = CreateDIBSection(screenDc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    ReleaseDC(nullptr, screenDc);

    if (hBmp && bits) {
        std::memset(bits, 0, sz * sz * 4);
        // Bottom-up DIB with straight (non-premultiplied) alpha, the layout LoadImage
        // produces for a 32bpp BMP.
        BYTE* lastRow = static_cast<BYTE*>(bits) + (sz - 1) * sz * 4;
        Gdiplus::Bitmap bmp(sz, sz, -sz * 4, PixelFormat32bppARGB, lastRow);
        Gdiplus::Graphics g(&bmp);
        g.Clear(Gdiplus::Color(0, 0, 0, 0));
        DrawSignInOptionGlyph(g);
    }

    Gdiplus::GdiplusShutdown(token);
    return hBmp;
}

}  // namespace FaceUnlock

extern "C" __declspec(dllexport) void WINAPI PreviewFaceIdHud(BOOL success) {
    FaceUnlock::FaceIdHud hud;
    hud.StartScanning();
    Sleep(850);
    hud.FinishWithResult(success != FALSE);
}

extern "C" __declspec(dllexport) void* WINAPI FaceIdHudStart() {
    auto* hud = new (std::nothrow) FaceUnlock::FaceIdHud();
    if (hud) {
        hud->StartScanning();
    }
    return hud;
}

extern "C" __declspec(dllexport) void WINAPI FaceIdHudFinish(void* handle, BOOL success) {
    if (!handle) return;
    auto* hud = static_cast<FaceUnlock::FaceIdHud*>(handle);
    hud->FinishWithResult(success != FALSE);
    delete hud;
}
