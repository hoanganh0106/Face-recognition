#include "FaceCredential.h"
#include "FaceCredentialProvider.h"
#include "FaceIdHud.h"
#include "NativeFaceEngine.h"
#include "helpers.h"
#include "guid.h"
#include <shlwapi.h>

namespace FaceUnlock {

namespace {
constexpr UINT_PTR kRetryTimerId = 0xFAce1D01;
constexpr UINT kRetryPollIntervalMs = 40;
}

static HRESULT AllocStr(PCWSTR src, PWSTR* dst) {
    size_t cch = wcslen(src) + 1;
    *dst = (PWSTR)CoTaskMemAlloc(cch * sizeof(WCHAR));
    if (!*dst) return E_OUTOFMEMORY;
    wcscpy_s(*dst, cch, src);
    return S_OK;
}

FaceCredential::FaceCredential()
    : m_cRef(1),
      m_cpus(CPUS_INVALID),
      m_pProvider(nullptr),
      m_pEvents(nullptr),
      m_label(L"Face ID"),
      m_status(L"Đang nhận diện khuôn mặt..."),
      m_rescanLabel(L"Quét lại Face ID"),
      m_autoTriggered(false),
      m_selected(false),
      m_inScan(false),
      m_waitingForRetry(false),
      m_hwndRetry(nullptr),
      m_prevSpaceDown(false),
      m_prevEnterDown(false),
      m_prevFDown(false),
      m_prevCtrlDown(false),
      m_lastCtrlTapTick(0),
      m_lastScanEndTick(0) {}

FaceCredential::~FaceCredential() {
    StopRetryWatch();
    if (m_pEvents) {
        m_pEvents->Release();
        m_pEvents = nullptr;
    }
}

HRESULT FaceCredential::Initialize(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus,
                                   FaceCredentialProvider* ownerProvider) {
    m_cpus = cpus;
    m_pProvider = ownerProvider;
    m_autoTriggered = false;
    return S_OK;
}

void FaceCredential::PrepareImmediateRescan() noexcept {
    m_autoTriggered = false;
    m_waitingForRetry = false;
    m_status = L"Đang nhận diện khuôn mặt...";
}

IFACEMETHODIMP FaceCredential::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown || riid == IID_ICredentialProviderCredential) {
        *ppv = static_cast<ICredentialProviderCredential*>(this);
        AddRef();
        return S_OK;
    }
    return E_NOINTERFACE;
}

IFACEMETHODIMP_(ULONG) FaceCredential::AddRef()  { return InterlockedIncrement(&m_cRef); }
IFACEMETHODIMP_(ULONG) FaceCredential::Release() { LONG c = InterlockedDecrement(&m_cRef); if (c == 0) delete this; return c; }

IFACEMETHODIMP FaceCredential::Advise(ICredentialProviderCredentialEvents* e) {
    if (m_pEvents) m_pEvents->Release();
    m_pEvents = e;
    if (e) e->AddRef();
    return S_OK;
}

IFACEMETHODIMP FaceCredential::UnAdvise() {
    StopRetryWatch();
    if (m_pEvents) {
        m_pEvents->Release();
        m_pEvents = nullptr;
    }
    return S_OK;
}

IFACEMETHODIMP FaceCredential::SetSelected(BOOL* pbAutoLogon) {
    m_selected = true;
    if (!m_autoTriggered) {
        m_autoTriggered = true;
        m_waitingForRetry = false;
        StopRetryWatch();
        *pbAutoLogon = TRUE;
        m_status = L"Đang nhận diện khuôn mặt...";
    } else {
        *pbAutoLogon = FALSE;
        m_status = L"Nhấn Space hoặc Enter để quét lại";
        StartRetryWatch();
    }
    if (m_pEvents) {
        m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
    }
    return S_OK;
}

IFACEMETHODIMP FaceCredential::SetDeselected() {
    m_selected = false;
    m_autoTriggered = false;
    m_waitingForRetry = false;
    StopRetryWatch();
    return S_OK;
}

IFACEMETHODIMP FaceCredential::GetFieldState(DWORD dwFieldID,
                                             CREDENTIAL_PROVIDER_FIELD_STATE* pcpfs,
                                             CREDENTIAL_PROVIDER_FIELD_INTERACTIVE_STATE* pcpfis) {
    if (dwFieldID >= FIELD_COUNT) return E_INVALIDARG;
    if (dwFieldID == FIELD_RESCAN) {
        *pcpfs  = m_waitingForRetry ? CPFS_DISPLAY_IN_SELECTED_TILE : CPFS_HIDDEN;
        *pcpfis = m_waitingForRetry ? CPFIS_FOCUSED : CPFIS_NONE;
        return S_OK;
    }
    *pcpfs  = s_FieldStatePairs[dwFieldID].cpfs;
    *pcpfis = s_FieldStatePairs[dwFieldID].cpfis;
    return S_OK;
}

IFACEMETHODIMP FaceCredential::GetStringValue(DWORD dwFieldID, PWSTR* ppwsz) {
    switch (dwFieldID) {
        case FIELD_LABEL:  return AllocStr(m_label.c_str(),       ppwsz);
        case FIELD_STATUS: return AllocStr(m_status.c_str(),      ppwsz);
        case FIELD_RESCAN: return AllocStr(m_rescanLabel.c_str(), ppwsz);
    }
    return E_INVALIDARG;
}

IFACEMETHODIMP FaceCredential::GetBitmapValue(DWORD dwFieldID, HBITMAP* phbmp) {
    if (dwFieldID != FIELD_TILE_IMAGE || !phbmp) return E_INVALIDARG;
    HBITMAP hBmp = CreateFaceIdTileBitmap();
    if (!hBmp) {
        *phbmp = nullptr;
        return E_FAIL;
    }
    *phbmp = hBmp;
    return S_OK;
}

IFACEMETHODIMP FaceCredential::GetSubmitButtonValue(DWORD dwFieldID, DWORD* pdwAdjacentTo) {
    if (dwFieldID != FIELD_SUBMIT) return E_INVALIDARG;
    *pdwAdjacentTo = FIELD_STATUS;
    return S_OK;
}

IFACEMETHODIMP FaceCredential::CommandLinkClicked(DWORD dwFieldID) {
    if (dwFieldID == FIELD_RESCAN) {
        TriggerRescan();
        return S_OK;
    }
    return E_NOTIMPL;
}

LRESULT CALLBACK FaceCredential::RetryMsgWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        return TRUE;
    }
    auto* self = reinterpret_cast<FaceCredential*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (msg == WM_TIMER && wParam == kRetryTimerId && self) {
        self->PollRetryKeys();
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

void FaceCredential::StartRetryWatch() noexcept {
    m_waitingForRetry = true;
    // Snapshot current key states so keys already held down while waking screen do not misfire
    m_prevSpaceDown = (GetAsyncKeyState(VK_SPACE)   & 0x8000) != 0;
    m_prevEnterDown = (GetAsyncKeyState(VK_RETURN)  & 0x8000) != 0;
    m_prevFDown     = (GetAsyncKeyState('F')        & 0x8000) != 0;
    m_prevCtrlDown  = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;

    if (!m_hwndRetry) {
        const wchar_t* cls = L"FaceUnlockStaRetryMsgWnd";
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = RetryMsgWndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = cls;
        RegisterClassExW(&wc);

        m_hwndRetry = CreateWindowExW(0, cls, L"", 0, 0, 0, 0, 0,
                                      HWND_MESSAGE, nullptr, wc.hInstance, this);
    }
    if (m_hwndRetry) {
        SetTimer(m_hwndRetry, kRetryTimerId, kRetryPollIntervalMs, nullptr);
    }
}

void FaceCredential::StopRetryWatch() noexcept {
    if (m_hwndRetry) {
        KillTimer(m_hwndRetry, kRetryTimerId);
        DestroyWindow(m_hwndRetry);
        m_hwndRetry = nullptr;
    }
}

void FaceCredential::PollRetryKeys() noexcept {
    if (!m_selected || !m_waitingForRetry || m_inScan) return;

    ULONGLONG now = GetTickCount64();
    if (now - m_lastScanEndTick < 350ULL) {
        m_prevSpaceDown = (GetAsyncKeyState(VK_SPACE)   & 0x8000) != 0;
        m_prevEnterDown = (GetAsyncKeyState(VK_RETURN)  & 0x8000) != 0;
        m_prevFDown     = (GetAsyncKeyState('F')        & 0x8000) != 0;
        m_prevCtrlDown  = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
        return;
    }

    bool spaceDown = (GetAsyncKeyState(VK_SPACE)   & 0x8000) != 0;
    bool enterDown = (GetAsyncKeyState(VK_RETURN)  & 0x8000) != 0;
    bool fDown     = (GetAsyncKeyState('F')        & 0x8000) != 0;
    bool ctrlDown  = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;

    bool spaceEdge = spaceDown && !m_prevSpaceDown;
    bool enterEdge = enterDown && !m_prevEnterDown;
    bool fEdge     = fDown     && !m_prevFDown;
    bool ctrlDoubleTap = false;

    if (ctrlDown && !m_prevCtrlDown) {
        if (now - m_lastCtrlTapTick <= 420ULL) {
            ctrlDoubleTap = true;
            m_lastCtrlTapTick = 0;
        } else {
            m_lastCtrlTapTick = now;
        }
    }

    m_prevSpaceDown = spaceDown;
    m_prevEnterDown = enterDown;
    m_prevFDown     = fDown;
    m_prevCtrlDown  = ctrlDown;

    if (spaceEdge || enterEdge || fEdge || ctrlDoubleTap) {
        TriggerRescan();
    }
}

void FaceCredential::TriggerRescan() noexcept {
    ULONGLONG now = GetTickCount64();
    if (m_inScan || (now - m_lastScanEndTick < 350ULL)) return;
    StopRetryWatch();
    if (m_pProvider) {
        m_pProvider->RequestRescan();
    }
}

IFACEMETHODIMP FaceCredential::GetSerialization(
    CREDENTIAL_PROVIDER_GET_SERIALIZATION_RESPONSE* pcpgsr,
    CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* pcpcs,
    PWSTR* ppwszOptionalStatusText,
    CREDENTIAL_PROVIDER_STATUS_ICON* pcpsiOptionalStatusIcon) {

    *pcpgsr = CPGSR_NO_CREDENTIAL_NOT_FINISHED;
    *pcpsiOptionalStatusIcon = CPSI_NONE;
    if (ppwszOptionalStatusText) *ppwszOptionalStatusText = nullptr;

    m_inScan = true;
    m_waitingForRetry = false;
    StopRetryWatch();
    if (m_pEvents) {
        m_pEvents->SetFieldState(this, FIELD_RESCAN, CPFS_HIDDEN);
        m_pEvents->SetFieldString(this, FIELD_STATUS, L"Đang nhận diện khuôn mặt...");
    }

    // 100% Pure C++20 Zero-Daemon On-Demand Flow:
    // 1. Drop down 120–165 FPS ProMotion iOS Dynamic Island HUD
    // 2. Open webcam + run YuNet + MiniFASNet + SFace in parallel, shut down webcam LED immediately
    // 3. On mismatch: Dynamic Island simply flashes coral-red & shakes smoothly, then retracts —
    //    NO ugly Windows modal notification screen, NO square error icon!
    FaceIdHud hud;
    hud.StartScanning();

    std::wstring user, pw, dom;
    std::string err;
    bool ok = false;
    try {
        ok = NativeRequestUnlock(user, pw, dom, err);
    } catch (...) {
        ok = false;
        err = "cpp-exception";
    }

    hud.FinishWithResult(ok);
    m_inScan = false;
    m_lastScanEndTick = GetTickCount64();

    if (!ok) {
        if (!pw.empty()) SecureZeroMemory(pw.data(), pw.size() * sizeof(wchar_t));
        m_status = L"Nhấn Space hoặc Enter để quét lại";
        m_waitingForRetry = true;
        if (m_pEvents) {
            m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
            m_pEvents->SetFieldState(this, FIELD_RESCAN, CPFS_DISPLAY_IN_SELECTED_TILE);
            m_pEvents->SetFieldInteractiveState(this, FIELD_RESCAN, CPFIS_FOCUSED);
        }
        if (m_selected) {
            StartRetryWatch();
        }
        // Return S_OK with CPGSR_NO_CREDENTIAL_NOT_FINISHED + nullptr status text + CPSI_NONE
        // so Windows LogonUI stays right on the clean Lock Screen WITHOUT any error modal!
        return S_OK;
    }

    HRESULT hr = E_FAIL;
    try {
        hr = KerbPackInteractiveUnlock(dom, user, pw, m_cpus, pcpcs);
    } catch (...) {
        hr = E_FAIL;
    }
    if (!pw.empty()) {
        SecureZeroMemory(pw.data(), pw.size() * sizeof(wchar_t));
    }

    if (FAILED(hr)) {
        m_status = L"Lỗi đóng gói thông tin xác thực — chọn PIN/Mật khẩu";
        m_waitingForRetry = true;
        if (m_pEvents) {
            m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
            m_pEvents->SetFieldState(this, FIELD_RESCAN, CPFS_DISPLAY_IN_SELECTED_TILE);
        }
        return S_OK;
    }

    pcpcs->clsidCredentialProvider = CLSID_FaceCredentialProvider;
    *pcpgsr = CPGSR_RETURN_CREDENTIAL_FINISHED;
    return S_OK;
}

IFACEMETHODIMP FaceCredential::ReportResult(NTSTATUS ntsStatus, NTSTATUS ntsSubstatus,
                                            PWSTR* ppwszOptionalStatusText,
                                            CREDENTIAL_PROVIDER_STATUS_ICON* pcpsiOptionalStatusIcon) {
    (void)ntsStatus; (void)ntsSubstatus;
    *ppwszOptionalStatusText = nullptr;
    *pcpsiOptionalStatusIcon = CPSI_NONE;
    return S_OK;
}

}  // namespace FaceUnlock
