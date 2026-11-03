#pragma once
#include <windows.h>
#include <credentialprovider.h>
#include <string>

namespace FaceUnlock {

class FaceCredentialProvider;

class FaceCredential final : public ICredentialProviderCredential {
public:
    FaceCredential();
    ~FaceCredential();

    HRESULT Initialize(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus, FaceCredentialProvider* ownerProvider);
    void SetOwnerProvider(FaceCredentialProvider* ownerProvider) noexcept { m_pProvider = ownerProvider; }
    void PrepareImmediateRescan() noexcept;
    void StopRetryWatch() noexcept;

    // IUnknown
    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv) override;
    IFACEMETHODIMP_(ULONG) AddRef() override;
    IFACEMETHODIMP_(ULONG) Release() override;

    // ICredentialProviderCredential
    IFACEMETHODIMP Advise(ICredentialProviderCredentialEvents* pcpce) override;
    IFACEMETHODIMP UnAdvise() override;
    IFACEMETHODIMP SetSelected(BOOL* pbAutoLogon) override;
    IFACEMETHODIMP SetDeselected() override;
    IFACEMETHODIMP GetFieldState(DWORD dwFieldID,
                                 CREDENTIAL_PROVIDER_FIELD_STATE* pcpfs,
                                 CREDENTIAL_PROVIDER_FIELD_INTERACTIVE_STATE* pcpfis) override;
    IFACEMETHODIMP GetStringValue(DWORD dwFieldID, PWSTR* ppwsz) override;
    IFACEMETHODIMP GetBitmapValue(DWORD dwFieldID, HBITMAP* phbmp) override;
    IFACEMETHODIMP GetCheckboxValue(DWORD, BOOL*, PWSTR*) override { return E_NOTIMPL; }
    IFACEMETHODIMP GetSubmitButtonValue(DWORD dwFieldID, DWORD* pdwAdjacentTo) override;
    IFACEMETHODIMP GetComboBoxValueCount(DWORD, DWORD*, DWORD*) override { return E_NOTIMPL; }
    IFACEMETHODIMP GetComboBoxValueAt(DWORD, DWORD, PWSTR*) override { return E_NOTIMPL; }
    IFACEMETHODIMP SetStringValue(DWORD, PCWSTR) override { return S_OK; }
    IFACEMETHODIMP SetCheckboxValue(DWORD, BOOL) override { return E_NOTIMPL; }
    IFACEMETHODIMP SetComboBoxSelectedValue(DWORD, DWORD) override { return E_NOTIMPL; }
    IFACEMETHODIMP CommandLinkClicked(DWORD dwFieldID) override;
    IFACEMETHODIMP GetSerialization(CREDENTIAL_PROVIDER_GET_SERIALIZATION_RESPONSE* pcpgsr,
                                    CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* pcpcs,
                                    PWSTR* ppwszOptionalStatusText,
                                    CREDENTIAL_PROVIDER_STATUS_ICON* pcpsiOptionalStatusIcon) override;
    IFACEMETHODIMP ReportResult(NTSTATUS ntsStatus, NTSTATUS ntsSubstatus,
                                PWSTR* ppwszOptionalStatusText,
                                CREDENTIAL_PROVIDER_STATUS_ICON* pcpsiOptionalStatusIcon) override;

private:
    static LRESULT CALLBACK RetryMsgWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    void StartRetryWatch() noexcept;
    void PollRetryKeys() noexcept;
    void TriggerRescan() noexcept;

    LONG m_cRef;
    CREDENTIAL_PROVIDER_USAGE_SCENARIO m_cpus;
    FaceCredentialProvider* m_pProvider;
    ICredentialProviderCredentialEvents* m_pEvents;
    std::wstring m_label;
    std::wstring m_status;
    std::wstring m_rescanLabel;
    bool m_autoTriggered;
    bool m_selected;
    bool m_inScan;
    bool m_waitingForRetry;

    // STA message-only window & edge-triggered key states (active ONLY after failed scan while tile selected)
    HWND m_hwndRetry;
    bool m_prevSpaceDown;
    bool m_prevEnterDown;
    bool m_prevFDown;
    bool m_prevCtrlDown;
    ULONGLONG m_lastCtrlTapTick;
    ULONGLONG m_lastScanEndTick;
};

}  // namespace FaceUnlock
