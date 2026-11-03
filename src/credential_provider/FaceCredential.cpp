#include "FaceCredential.h"
#include "FaceIdHud.h"
#include "NativeFaceEngine.h"
#include "helpers.h"
#include "guid.h"
#include <shlwapi.h>

namespace FaceUnlock {

static HRESULT AllocStr(PCWSTR src, PWSTR* dst) {
    size_t cch = wcslen(src) + 1;
    *dst = (PWSTR)CoTaskMemAlloc(cch * sizeof(WCHAR));
    if (!*dst) return E_OUTOFMEMORY;
    wcscpy_s(*dst, cch, src);
    return S_OK;
}

FaceCredential::FaceCredential()
    : m_cRef(1), m_cpus(CPUS_INVALID), m_pEvents(nullptr),
      m_label(L"Face ID"), m_status(L"Đang nhận diện khuôn mặt..."),
      m_autoTriggered(false) {}

FaceCredential::~FaceCredential() {}

HRESULT FaceCredential::Initialize(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus) {
    m_cpus = cpus;
    m_autoTriggered = false;
    return S_OK;
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
    if (m_pEvents) { m_pEvents->Release(); m_pEvents = nullptr; }
    return S_OK;
}

IFACEMETHODIMP FaceCredential::SetSelected(BOOL* pbAutoLogon) {
    // Only triggers when user actively wakes/clicks the Lock Screen or selects the Face ID tile.
    if (!m_autoTriggered) {
        m_autoTriggered = true;
        *pbAutoLogon = TRUE;
        m_status = L"Đang nhận diện khuôn mặt...";
    } else {
        *pbAutoLogon = FALSE;
        m_status = L"Nhấn Enter hoặc mũi tên để quét lại";
    }
    if (m_pEvents) m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
    return S_OK;
}

IFACEMETHODIMP FaceCredential::SetDeselected() {
    m_autoTriggered = false;
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
    switch (dwFieldID) {
        case FIELD_LABEL:  return AllocStr(m_label.c_str(),  ppwsz);
        case FIELD_STATUS: return AllocStr(m_status.c_str(), ppwsz);
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

IFACEMETHODIMP FaceCredential::GetSerialization(
    CREDENTIAL_PROVIDER_GET_SERIALIZATION_RESPONSE* pcpgsr,
    CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* pcpcs,
    PWSTR* ppwszOptionalStatusText,
    CREDENTIAL_PROVIDER_STATUS_ICON* pcpsiOptionalStatusIcon) {

    *pcpgsr = CPGSR_NO_CREDENTIAL_NOT_FINISHED;
    *pcpsiOptionalStatusIcon = CPSI_NONE;
    if (ppwszOptionalStatusText) *ppwszOptionalStatusText = nullptr;

    // 100% Pure C++20 Zero-Daemon On-Demand Flow:
    // 1. Spawn 60-FPS iOS Dynamic Island HUD on Winlogon desktop
    // 2. Open webcam via MediaFoundation + load ONNX models in parallel
    // 3. Verify YuNet + MiniFASNet liveness + SFace embedding, immediately shut down webcam
    // 4. Play green checkmark (or red shake) and submit credentials to LSA
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

    if (!ok) {
        if (!pw.empty()) SecureZeroMemory(pw.data(), pw.size() * sizeof(wchar_t));
        m_status = L"Không nhận diện được. Nhấn Enter hoặc mũi tên để thử lại.";
        if (m_pEvents) m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
        *pcpsiOptionalStatusIcon = CPSI_ERROR;
        AllocStr(L"Không khớp khuôn mặt - Thử lại hoặc chọn PIN/Mật khẩu", ppwszOptionalStatusText);
        return S_FALSE;
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
        m_status = L"Lỗi đóng gói thông tin xác thực";
        if (m_pEvents) m_pEvents->SetFieldString(this, FIELD_STATUS, m_status.c_str());
        *pcpsiOptionalStatusIcon = CPSI_ERROR;
        return S_FALSE;
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
