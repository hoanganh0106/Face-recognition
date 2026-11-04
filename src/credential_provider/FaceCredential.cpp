#include "FaceCredential.h"
#include "FaceCredentialProvider.h"
#include "FaceIdHud.h"
#include "NativeFaceEngine.h"
#include "helpers.h"
#include "guid.h"
#include <shlwapi.h>
#include <uiautomation.h>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <new>

namespace FaceUnlock {

namespace {
constexpr UINT_PTR kRetryTimerId = 0xFAce1D01;
constexpr UINT kRetryPollIntervalMs = 40;
constexpr UINT kMsgClickProbeResult = WM_APP + 0x1D;
constexpr ULONGLONG kClickProbeTimeoutMs = 1500;
constexpr const wchar_t* kRetryHint = L"Chạm hoặc nhấn Space / Enter để quét lại Face ID";

// {ff48dba4-60ef-4201-aa87-54103eef594e}
const CLSID kClsidCUIAutomation = {
    0xff48dba4, 0x60ef, 0x4201, { 0xaa, 0x87, 0x54, 0x10, 0x3e, 0xef, 0x59, 0x4e }
};
// {30cbe57d-d9d0-452a-ab13-7ac5ac4825ee}
const IID kIidIUIAutomation = {
    0x30cbe57d, 0xd9d0, 0x452a, { 0xab, 0x13, 0x7a, 0xc5, 0xac, 0x48, 0x25, 0xee }
};

void LogEvent(const char* fmt, ...);

bool IsInteractiveControlType(CONTROLTYPEID type) {
    switch (type) {
        case UIA_ButtonControlTypeId:
        case UIA_SplitButtonControlTypeId:
        case UIA_HyperlinkControlTypeId:
        case UIA_EditControlTypeId:
        case UIA_ComboBoxControlTypeId:
        case UIA_CheckBoxControlTypeId:
        case UIA_RadioButtonControlTypeId:
        case UIA_MenuItemControlTypeId:
            return true;
        default:
            return false;
    }
}

// Checks the element under the point and its direct parent, so a click on the label
// inside a button still counts as a click on that button.
bool IsPointOnInteractiveControl(POINT pt) {
    HRESULT hrInit = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool onControl = false;

    IUIAutomation* uia = nullptr;
    if (SUCCEEDED(CoCreateInstance(kClsidCUIAutomation, nullptr, CLSCTX_INPROC_SERVER,
                                   kIidIUIAutomation, reinterpret_cast<void**>(&uia))) && uia) {
        IUIAutomationElement* element = nullptr;
        if (SUCCEEDED(uia->ElementFromPoint(pt, &element)) && element) {
            IUIAutomationTreeWalker* walker = nullptr;
            uia->get_ControlViewWalker(&walker);

            IUIAutomationElement* current = element;
            for (int depth = 0; current && depth < 2 && !onControl; ++depth) {
                CONTROLTYPEID type = 0;
                if (SUCCEEDED(current->get_CurrentControlType(&type)) && IsInteractiveControlType(type)) {
                    onControl = true;
                }
                BSTR name = nullptr;
                current->get_CurrentName(&name);
                char nameUtf8[128] = "";
                if (name) {
                    WideCharToMultiByte(CP_UTF8, 0, name, -1, nameUtf8, sizeof(nameUtf8), nullptr, nullptr);
                    SysFreeString(name);
                }
                LogEvent("Click probe depth=%d type=%ld name=\"%s\"", depth, static_cast<long>(type), nameUtf8);
                IUIAutomationElement* parent = nullptr;
                if (!onControl && walker) {
                    walker->GetParentElement(current, &parent);
                }
                current->Release();
                current = parent;
            }
            if (current) current->Release();
            if (walker) walker->Release();
        }
        uia->Release();
    }

    if (SUCCEEDED(hrInit)) CoUninitialize();
    return onControl;
}

void LogEvent(const char* fmt, ...) {
    const wchar_t* dir = L"C:\\ProgramData\\FaceUnlock\\logs";
    const wchar_t* path = L"C:\\ProgramData\\FaceUnlock\\logs\\credential.log";
    CreateDirectoryW(dir, nullptr);

    WIN32_FILE_ATTRIBUTE_DATA attr{};
    bool tooBig = GetFileAttributesExW(path, GetFileExInfoStandard, &attr) &&
                  (attr.nFileSizeHigh != 0 || attr.nFileSizeLow > 256u * 1024u);
    HANDLE file = CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              tooBig ? CREATE_ALWAYS : OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;

    SYSTEMTIME st{};
    GetLocalTime(&st);
    char line[512];
    int n = snprintf(line, sizeof(line), "%02u:%02u:%02u.%03u ",
                     st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    va_list args;
    va_start(args, fmt);
    int m = vsnprintf(line + n, sizeof(line) - n - 2, fmt, args);
    va_end(args);
    if (m > 0) n += (std::min)(m, static_cast<int>(sizeof(line)) - n - 3);
    line[n++] = '\r';
    line[n++] = '\n';

    DWORD written = 0;
    WriteFile(file, line, static_cast<DWORD>(n), &written, nullptr);
    CloseHandle(file);
}

struct ClickProbeRequest {
    HWND hwndReply;
    POINT pt;
    HMODULE module;
};

// Runs off the LogonUI thread: UI Automation calls back into LogonUI's own UI thread,
// so querying it synchronously from there would deadlock.
DWORD WINAPI ClickProbeThread(LPVOID param) {
    auto* req = static_cast<ClickProbeRequest*>(param);
    bool onControl = IsPointOnInteractiveControl(req->pt);
    PostMessageW(req->hwndReply, kMsgClickProbeResult, onControl ? 1 : 0, 0);
    HMODULE module = req->module;
    delete req;
    if (module) {
        FreeLibraryAndExitThread(module, 0);
    }
    return 0;
}
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
      m_userSid(),
      m_status(L""),
      m_autoTriggered(false),
      m_selected(false),
      m_inScan(false),
      m_waitingForRetry(false),
      m_waitingForWake(false),
      m_hasScanned(false),
      m_rescanPending(false),
      m_prevKeys{},
      m_hwndRetry(nullptr),
      m_prevSpaceDown(false),
      m_prevEnterDown(false),
      m_prevClickDown(false),
      m_clickProbePending(false),
      m_clickProbeTick(0),
      m_lastScanEndTick(0) {}

FaceCredential::~FaceCredential() {
    StopRetryWatch();
    m_hud.StopImmediate();
    if (m_pEvents) {
        m_pEvents->Release();
        m_pEvents = nullptr;
    }
}

HRESULT FaceCredential::Initialize(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus,
                                   FaceCredentialProvider* ownerProvider,
                                   const std::wstring& userSid) {
    m_cpus = cpus;
    m_pProvider = ownerProvider;
    m_userSid = userSid;
    m_autoTriggered = false;
    return S_OK;
}

void FaceCredential::UpdateContext(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus,
                                   const std::wstring& userSid) {
    if (cpus != CPUS_INVALID) {
        m_cpus = cpus;
    }
    if (!userSid.empty()) {
        m_userSid = userSid;
    }
}

void FaceCredential::PrepareImmediateRescan() noexcept {
    m_autoTriggered = false;
    m_waitingForRetry = false;
    m_rescanPending = true;
    m_status = L"";
}

IFACEMETHODIMP FaceCredential::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown ||
        riid == IID_ICredentialProviderCredential ||
        riid == IID_ICredentialProviderCredential2) {
        *ppv = static_cast<ICredentialProviderCredential2*>(this);
        AddRef();
        return S_OK;
    }
    return E_NOINTERFACE;
}

IFACEMETHODIMP_(ULONG) FaceCredential::AddRef()  { return InterlockedIncrement(&m_cRef); }
IFACEMETHODIMP_(ULONG) FaceCredential::Release() { LONG c = InterlockedDecrement(&m_cRef); if (c == 0) delete this; return c; }

IFACEMETHODIMP FaceCredential::GetUserSid(PWSTR* sid) {
    if (!sid) return E_POINTER;
    *sid = nullptr;
    if (m_userSid.empty()) {
        return S_FALSE;
    }
    return AllocStr(m_userSid.c_str(), sid);
}

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
    LogEvent("SetSelected autoTriggered=%d rescanPending=%d waitingForWake=%d hasScanned=%d",
             m_autoTriggered, m_rescanPending, m_waitingForWake, m_hasScanned);

    if (m_rescanPending) {
        m_autoTriggered = true;
        m_waitingForRetry = false;
        StopRetryWatch();
        *pbAutoLogon = TRUE;
        m_status = L"";
    } else if (!m_hasScanned) {
        // LogonUI selects the tile right after Win+L, behind the lock screen curtain, and
        // ignores auto-logon there. Until the first scan, the input that dismisses the
        // curtain (any click or key) starts it. LogonUI also re-selects the tile after a
        // failed scan; from then on only empty-area clicks or Space/Enter rescan, so the
        // Sign-in options link stays usable.
        m_autoTriggered = true;
        m_waitingForWake = true;
        *pbAutoLogon = TRUE;
        m_status = L"";
        StartRetryWatch();
    } else {
        *pbAutoLogon = FALSE;
        m_status = kRetryHint;
        StartRetryWatch();
    }
    if (m_pEvents) {
        m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
    }
    return S_OK;
}

IFACEMETHODIMP FaceCredential::SetDeselected() {
    LogEvent("SetDeselected rescanPending=%d", m_rescanPending);
    m_selected = false;
    m_autoTriggered = false;
    m_waitingForRetry = false;
    m_waitingForWake = false;
    StopRetryWatch();
    // LogonUI may deselect while re-enumerating for a requested rescan; keep the island
    // on screen so it switches straight back to scanning instead of retracting.
    if (!m_rescanPending) {
        m_hud.Hide();
    }
    return S_OK;
}

IFACEMETHODIMP FaceCredential::GetFieldState(DWORD dwFieldID,
                                             CREDENTIAL_PROVIDER_FIELD_STATE* pcpfs,
                                             CREDENTIAL_PROVIDER_FIELD_INTERACTIVE_STATE* pcpfis) {
    if (dwFieldID >= FIELD_COUNT) return E_INVALIDARG;
    *pcpfs  = s_FieldStatePairs[dwFieldID].cpfs;
    *pcpfis = s_FieldStatePairs[dwFieldID].cpfis;
    return S_OK;
}

IFACEMETHODIMP FaceCredential::GetStringValue(DWORD dwFieldID, PWSTR* ppwsz) {
    if (dwFieldID == FIELD_STATUS) {
        return AllocStr(m_status.c_str(), ppwsz);
    }
    return E_INVALIDARG;
}

IFACEMETHODIMP FaceCredential::GetBitmapValue(DWORD dwFieldID, HBITMAP* phbmp) {
    if (!phbmp) return E_POINTER;
    *phbmp = nullptr;
    if (dwFieldID != FIELD_TILEIMAGE) return E_INVALIDARG;
    LogEvent("GetBitmapValue: served transparent bitmap");
    *phbmp = CreateFaceIdSignInOptionBitmap();
    return *phbmp ? S_OK : E_OUTOFMEMORY;
}

IFACEMETHODIMP FaceCredential::GetSubmitButtonValue(DWORD dwFieldID, DWORD* pdwAdjacentTo) {
    if (dwFieldID != FIELD_SUBMIT) return E_INVALIDARG;
    *pdwAdjacentTo = FIELD_STATUS;
    return S_OK;
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
    if (msg == kMsgClickProbeResult && self) {
        self->m_clickProbePending = false;
        LogEvent("Click probe onControl=%d", static_cast<int>(wParam));
        if (wParam == 0 && self->m_selected && self->m_waitingForRetry) {
            self->TriggerRescan("click");
        }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static bool IsPrimaryMouseDown() {
    int vk = GetSystemMetrics(SM_SWAPBUTTON) ? VK_RBUTTON : VK_LBUTTON;
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

void FaceCredential::StartRetryWatch() noexcept {
    m_waitingForRetry = true;
    m_prevSpaceDown = (GetAsyncKeyState(VK_SPACE)   & 0x8000) != 0;
    m_prevEnterDown = (GetAsyncKeyState(VK_RETURN)  & 0x8000) != 0;
    m_prevClickDown = IsPrimaryMouseDown();
    m_clickProbePending = false;
    for (int vk = 0; vk < 256; ++vk) {
        m_prevKeys[vk] = (GetAsyncKeyState(vk) & 0x8000) != 0;
    }

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
    if (m_clickProbePending && now - m_clickProbeTick > kClickProbeTimeoutMs) {
        m_clickProbePending = false;
    }

    bool spaceDown = (GetAsyncKeyState(VK_SPACE)   & 0x8000) != 0;
    bool enterDown = (GetAsyncKeyState(VK_RETURN)  & 0x8000) != 0;
    bool clickDown = IsPrimaryMouseDown();

    bool spaceEdge = spaceDown && !m_prevSpaceDown;
    bool enterEdge = enterDown && !m_prevEnterDown;
    bool clickEdge = clickDown && !m_prevClickDown;

    m_prevSpaceDown = spaceDown;
    m_prevEnterDown = enterDown;
    m_prevClickDown = clickDown;

    int wakeKey = -1;
    if (m_waitingForWake) {
        for (int vk = 0x08; vk < 256; ++vk) {
            bool down = (GetAsyncKeyState(vk) & 0x8000) != 0;
            if (down && !m_prevKeys[vk] && wakeKey < 0) wakeKey = vk;
            m_prevKeys[vk] = down;
        }
    }

    if (now - m_lastScanEndTick < 350ULL) return;

    if (m_waitingForWake && (clickEdge || wakeKey >= 0)) {
        char reason[32] = "wake-click";
        if (!clickEdge) snprintf(reason, sizeof(reason), "wake-key 0x%02x", wakeKey);
        TriggerRescan(reason);
    } else if (spaceEdge || enterEdge) {
        TriggerRescan(spaceEdge ? "space" : "enter");
    } else if (clickEdge && !m_clickProbePending) {
        POINT pt{};
        if (GetCursorPos(&pt)) {
            StartClickProbe(pt);
        } else {
            TriggerRescan("click-no-cursor");
        }
    }
}

void FaceCredential::StartClickProbe(POINT pt) noexcept {
    auto* req = new (std::nothrow) ClickProbeRequest{ m_hwndRetry, pt, nullptr };
    if (!req || !m_hwndRetry) {
        delete req;
        TriggerRescan("click-probe-alloc-failed");
        return;
    }
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                       reinterpret_cast<LPCWSTR>(&ClickProbeThread), &req->module);

    HANDLE thread = CreateThread(nullptr, 0, ClickProbeThread, req, 0, nullptr);
    if (!thread) {
        if (req->module) FreeLibrary(req->module);
        delete req;
        TriggerRescan("click-probe-thread-failed");
        return;
    }
    CloseHandle(thread);
    m_clickProbePending = true;
    m_clickProbeTick = GetTickCount64();
}

void FaceCredential::TriggerRescan(const char* reason) noexcept {
    ULONGLONG now = GetTickCount64();
    if (m_inScan || (now - m_lastScanEndTick < 350ULL)) return;
    LogEvent("Rescan requested (%s)", reason);
    m_waitingForWake = false;
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

    LogEvent("GetSerialization: scan start");
    m_inScan = true;
    m_hasScanned = true;
    m_waitingForRetry = false;
    m_waitingForWake = false;
    m_rescanPending = false;
    StopRetryWatch();
    m_status = L"";
    if (m_pEvents) {
        m_pEvents->SetFieldString(this, FIELD_STATUS, L"");
    }

    m_hud.StartScanning();

    std::wstring user, pw, dom;
    std::string err;
    bool ok = false;
    try {
        ok = NativeRequestUnlock(user, pw, dom, err);
    } catch (...) {
        ok = false;
        err = "cpp-exception";
    }

    LogEvent("GetSerialization: scan end ok=%d err=%s", ok, err.c_str());
    m_hud.FinishWithResult(ok, true);
    LogEvent("GetSerialization: island finished");
    m_inScan = false;
    m_lastScanEndTick = GetTickCount64();

    if (!ok) {
        if (!pw.empty()) SecureZeroMemory(pw.data(), pw.size() * sizeof(wchar_t));
        m_status = kRetryHint;
        m_waitingForRetry = true;
        if (m_pEvents) {
            m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
        }
        if (m_selected) {
            StartRetryWatch();
        }
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
        LogEvent("GetSerialization: pack failed hr=0x%08lx", static_cast<unsigned long>(hr));
        m_status = L"Lỗi xác thực — hãy chọn PIN/Mật khẩu bên dưới";
        m_waitingForRetry = true;
        if (m_pEvents) {
            m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
        }
        return S_OK;
    }

    pcpcs->clsidCredentialProvider = CLSID_FaceCredentialProvider;
    *pcpgsr = CPGSR_RETURN_CREDENTIAL_FINISHED;
    LogEvent("GetSerialization: credential returned");
    return S_OK;
}

IFACEMETHODIMP FaceCredential::ReportResult(NTSTATUS ntsStatus, NTSTATUS ntsSubstatus,
                                            PWSTR* ppwszOptionalStatusText,
                                            CREDENTIAL_PROVIDER_STATUS_ICON* pcpsiOptionalStatusIcon) {
    LogEvent("ReportResult status=0x%08lx substatus=0x%08lx",
             static_cast<unsigned long>(ntsStatus), static_cast<unsigned long>(ntsSubstatus));
    *ppwszOptionalStatusText = nullptr;
    *pcpsiOptionalStatusIcon = CPSI_NONE;
    return S_OK;
}

}  // namespace FaceUnlock
