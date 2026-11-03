#include "FaceCredentialProvider.h"
#include "FaceCredential.h"
#include "NativeFaceEngine.h"
#include "helpers.h"
#include <sddl.h>
#include <cwctype>
#include <new>

namespace FaceUnlock {

namespace {

// PKEY_Identity_UserName: {C4322503-78CA-49C6-9ACC-A68E2AFD7B6B}, pid 100
const PROPERTYKEY kPkeyIdentityUserName = {
    { 0xc4322503, 0x78ca, 0x49c6, { 0x9a, 0xcc, 0xa6, 0x8e, 0x2a, 0xfd, 0x7b, 0x6b } },
    100
};

std::wstring ReadLastLoggedOnUserSid() {
    HKEY hKey = nullptr;
    const wchar_t* sub = L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Authentication\\LogonUI";
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, sub, 0, KEY_READ, &hKey) != ERROR_SUCCESS) {
        return L"";
    }
    wchar_t buf[256]{};
    DWORD cb = sizeof(buf);
    DWORD type = 0;
    std::wstring out;
    if (RegQueryValueExW(hKey, L"SelectedUserSID", nullptr, &type, reinterpret_cast<BYTE*>(buf), &cb) == ERROR_SUCCESS &&
        type == REG_SZ && buf[0] != L'\0') {
        out = buf;
    } else {
        cb = sizeof(buf);
        if (RegQueryValueExW(hKey, L"LastLoggedOnUserSID", nullptr, &type, reinterpret_cast<BYTE*>(buf), &cb) == ERROR_SUCCESS &&
            type == REG_SZ && buf[0] != L'\0') {
            out = buf;
        }
    }
    RegCloseKey(hKey);
    return out;
}

std::wstring ResolveSamAccountFromSid(const wchar_t* sidStr) {
    if (!sidStr || !sidStr[0]) return L"";
    PSID pSid = nullptr;
    if (!ConvertStringSidToSidW(sidStr, &pSid) || !pSid) {
        return L"";
    }
    wchar_t name[256]{};
    wchar_t dom[256]{};
    DWORD cchName = 256;
    DWORD cchDom  = 256;
    SID_NAME_USE use = SidTypeUnknown;
    std::wstring res;
    if (LookupAccountSidW(nullptr, pSid, name, &cchName, dom, &cchDom, &use)) {
        res = name;
    }
    LocalFree(pSid);
    return res;
}

bool IsSystemOrSandboxAccount(const std::wstring& samName) {
    if (samName.empty()) return false;
    if (_wcsnicmp(samName.c_str(), L"CodexSandbox", 12) == 0) return true;
    if (_wcsicmp(samName.c_str(), L"WDAGUtilityAccount") == 0) return true;
    if (_wcsicmp(samName.c_str(), L"DefaultAccount") == 0) return true;
    if (_wcsicmp(samName.c_str(), L"Guest") == 0) return true;
    return false;
}

}  // namespace

FaceCredentialProvider::FaceCredentialProvider()
    : m_cRef(1),
      m_cpus(CPUS_INVALID),
      m_pCred(nullptr),
      m_pProviderEvents(nullptr),
      m_upAdviseContext(0),
      m_targetUserSid(),
      m_forceAutoLogonNext(false) {}

FaceCredentialProvider::~FaceCredentialProvider() {
    if (m_pProviderEvents) {
        m_pProviderEvents->Release();
        m_pProviderEvents = nullptr;
    }
    if (m_pCred) {
        m_pCred->SetOwnerProvider(nullptr);
        m_pCred->Release();
        m_pCred = nullptr;
    }
}

IFACEMETHODIMP FaceCredentialProvider::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == IID_IUnknown || riid == IID_ICredentialProvider) {
        *ppv = static_cast<ICredentialProvider*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == IID_ICredentialProviderSetUserArray) {
        *ppv = static_cast<ICredentialProviderSetUserArray*>(this);
        AddRef();
        return S_OK;
    }
    return E_NOINTERFACE;
}

IFACEMETHODIMP_(ULONG) FaceCredentialProvider::AddRef()  { return InterlockedIncrement(&m_cRef); }
IFACEMETHODIMP_(ULONG) FaceCredentialProvider::Release() { LONG c = InterlockedDecrement(&m_cRef); if (c == 0) delete this; return c; }

HRESULT FaceCredentialProvider::EnsureCredentialCreated() {
    if (!m_pCred) {
        m_pCred = new (std::nothrow) FaceCredential();
        if (!m_pCred) return E_OUTOFMEMORY;
        HRESULT hr = m_pCred->Initialize(m_cpus, this, m_targetUserSid);
        if (FAILED(hr)) {
            m_pCred->Release();
            m_pCred = nullptr;
            return hr;
        }
    } else {
        m_pCred->UpdateContext(m_cpus, m_targetUserSid);
    }
    return S_OK;
}

IFACEMETHODIMP FaceCredentialProvider::SetUsageScenario(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus, DWORD) {
    switch (cpus) {
        case CPUS_LOGON:
        case CPUS_UNLOCK_WORKSTATION:
            m_cpus = cpus;
            return EnsureCredentialCreated();
        default:
            return E_NOTIMPL;
    }
}

IFACEMETHODIMP FaceCredentialProvider::SetUserArray(ICredentialProviderUserArray* users) {
    if (!users) return E_INVALIDARG;

    DWORD count = 0;
    if (FAILED(users->GetCount(&count)) || count == 0) {
        return S_OK;
    }

    VaultInfo vinfo = NativeQueryVaultInfo();
    std::wstring lastSid = ReadLastLoggedOnUserSid();

    std::wstring exactVaultMatchSid;
    std::wstring lastLoggedMatchSid;
    std::wstring firstHumanSid;
    std::wstring firstAnySid;

    for (DWORD i = 0; i < count; ++i) {
        ICredentialProviderUser* pUser = nullptr;
        if (FAILED(users->GetAt(i, &pUser)) || !pUser) continue;

        PWSTR pszSid = nullptr;
        if (SUCCEEDED(pUser->GetSid(&pszSid)) && pszSid) {
            std::wstring sid(pszSid);
            CoTaskMemFree(pszSid);

            if (firstAnySid.empty()) {
                firstAnySid = sid;
            }

            std::wstring samName = ResolveSamAccountFromSid(sid.c_str());
            if (firstHumanSid.empty() && !IsSystemOrSandboxAccount(samName)) {
                firstHumanSid = sid;
            }

            PWSTR pszIdentityName = nullptr;
            std::wstring idName;
            if (SUCCEEDED(pUser->GetStringValue(kPkeyIdentityUserName, &pszIdentityName)) && pszIdentityName) {
                idName = pszIdentityName;
                CoTaskMemFree(pszIdentityName);
            }

            if (!vinfo.username.empty()) {
                if ((!idName.empty() && _wcsicmp(idName.c_str(), vinfo.username.c_str()) == 0) ||
                    (!samName.empty() && _wcsicmp(samName.c_str(), vinfo.username.c_str()) == 0)) {
                    exactVaultMatchSid = sid;
                }
            }

            if (!lastSid.empty() && _wcsicmp(sid.c_str(), lastSid.c_str()) == 0) {
                lastLoggedMatchSid = sid;
            }
        }
        pUser->Release();
    }

    if (!exactVaultMatchSid.empty()) {
        m_targetUserSid = exactVaultMatchSid;
    } else if (!lastLoggedMatchSid.empty()) {
        m_targetUserSid = lastLoggedMatchSid;
    } else if (!firstHumanSid.empty()) {
        m_targetUserSid = firstHumanSid;
    } else {
        m_targetUserSid = firstAnySid;
    }

    if (m_pCred) {
        m_pCred->UpdateContext(m_cpus, m_targetUserSid);
    }
    return S_OK;
}

IFACEMETHODIMP FaceCredentialProvider::SetSerialization(const CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION*) {
    return E_NOTIMPL;
}

IFACEMETHODIMP FaceCredentialProvider::Advise(ICredentialProviderEvents* pcpe, UINT_PTR upAdviseContext) {
    if (m_pProviderEvents) {
        m_pProviderEvents->Release();
        m_pProviderEvents = nullptr;
    }
    m_pProviderEvents = pcpe;
    m_upAdviseContext = upAdviseContext;
    if (m_pProviderEvents) {
        m_pProviderEvents->AddRef();
    }
    return S_OK;
}

IFACEMETHODIMP FaceCredentialProvider::UnAdvise() {
    if (m_pCred) {
        m_pCred->StopRetryWatch();
    }
    if (m_pProviderEvents) {
        m_pProviderEvents->Release();
        m_pProviderEvents = nullptr;
    }
    m_upAdviseContext = 0;
    return S_OK;
}

void FaceCredentialProvider::RequestRescan() {
    if (!m_pProviderEvents || !m_pCred) return;
    m_pCred->PrepareImmediateRescan();
    m_forceAutoLogonNext = true;
    m_pProviderEvents->CredentialsChanged(m_upAdviseContext);
}

IFACEMETHODIMP FaceCredentialProvider::GetFieldDescriptorCount(DWORD* pdwCount) {
    *pdwCount = FIELD_COUNT;
    return S_OK;
}

IFACEMETHODIMP FaceCredentialProvider::GetFieldDescriptorAt(DWORD dwIndex,
                                                            CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR** ppcpfd) {
    if (dwIndex >= FIELD_COUNT) return E_INVALIDARG;
    CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR src = s_FieldDescriptors[dwIndex];
    auto* out = (CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR*)CoTaskMemAlloc(sizeof(src));
    if (!out) return E_OUTOFMEMORY;
    *out = src;
    size_t cch = wcslen(src.pszLabel) + 1;
    out->pszLabel = (PWSTR)CoTaskMemAlloc(cch * sizeof(WCHAR));
    if (!out->pszLabel) { CoTaskMemFree(out); return E_OUTOFMEMORY; }
    wcscpy_s(out->pszLabel, cch, src.pszLabel);
    *ppcpfd = out;
    return S_OK;
}

IFACEMETHODIMP FaceCredentialProvider::GetCredentialCount(DWORD* pdwCount, DWORD* pdwDefault,
                                                          BOOL* pbAutoLogonWithDefault) {
    *pdwCount = m_pCred ? 1 : 0;
    *pdwDefault = 0;
    *pbAutoLogonWithDefault = m_forceAutoLogonNext ? TRUE : FALSE;
    m_forceAutoLogonNext = false;
    return S_OK;
}

IFACEMETHODIMP FaceCredentialProvider::GetCredentialAt(DWORD dwIndex, ICredentialProviderCredential** ppcpc) {
    if (dwIndex != 0 || !m_pCred) return E_INVALIDARG;
    return m_pCred->QueryInterface(IID_ICredentialProviderCredential, (void**)ppcpc);
}

}  // namespace FaceUnlock
