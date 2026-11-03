#pragma once
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NTDDI_VERSION
#define NTDDI_VERSION 0x0A000000
#endif
#include <windows.h>
#include <credentialprovider.h>
#include <string>

namespace FaceUnlock {

class FaceCredential;

// V2 User-Associated Credential Provider:
// Implements ICredentialProviderSetUserArray so Face ID attaches directly to the user's
// real Windows 11 account (preserving their native circular Windows 11 avatar & name,
// never spawning a legacy V1 square-icon tile).
class FaceCredentialProvider final : public ICredentialProvider,
                                     public ICredentialProviderSetUserArray {
public:
    FaceCredentialProvider();
    ~FaceCredentialProvider();

    // IUnknown
    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv) override;
    IFACEMETHODIMP_(ULONG) AddRef() override;
    IFACEMETHODIMP_(ULONG) Release() override;

    // ICredentialProvider
    IFACEMETHODIMP SetUsageScenario(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus, DWORD dwFlags) override;
    IFACEMETHODIMP SetSerialization(const CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION*) override;
    IFACEMETHODIMP Advise(ICredentialProviderEvents* pcpe, UINT_PTR upAdviseContext) override;
    IFACEMETHODIMP UnAdvise() override;
    IFACEMETHODIMP GetFieldDescriptorCount(DWORD* pdwCount) override;
    IFACEMETHODIMP GetFieldDescriptorAt(DWORD dwIndex,
                                        CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR** ppcpfd) override;
    IFACEMETHODIMP GetCredentialCount(DWORD* pdwCount, DWORD* pdwDefault,
                                      BOOL* pbAutoLogonWithDefault) override;
    IFACEMETHODIMP GetCredentialAt(DWORD dwIndex, ICredentialProviderCredential** ppcpc) override;

    // ICredentialProviderSetUserArray
    IFACEMETHODIMP SetUserArray(ICredentialProviderUserArray* users) override;

    // Triggered on STA thread when user presses Space / Enter / F / Double-Ctrl to rescan
    void RequestRescan();

private:
    HRESULT EnsureCredentialCreated();

    LONG m_cRef;
    CREDENTIAL_PROVIDER_USAGE_SCENARIO m_cpus;
    FaceCredential* m_pCred;
    ICredentialProviderEvents* m_pProviderEvents;
    UINT_PTR m_upAdviseContext;
    std::wstring m_targetUserSid;
    bool m_forceAutoLogonNext;
};

}  // namespace FaceUnlock
