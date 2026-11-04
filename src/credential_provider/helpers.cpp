#include "helpers.h"
#include <wincred.h>
#include <intsafe.h>

namespace FaceUnlock {

const CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR s_FieldDescriptors[FIELD_COUNT] = {
    { FIELD_SUBMIT, CPFT_SUBMIT_BUTTON, const_cast<PWSTR>(L"Face ID"), GUID_NULL },
    { FIELD_STATUS, CPFT_SMALL_TEXT,    const_cast<PWSTR>(L"Face ID"), GUID_NULL },
    { FIELD_TILEIMAGE, CPFT_TILE_IMAGE, const_cast<PWSTR>(L"Face ID"), GUID_NULL },
};

const FIELD_STATE_PAIR s_FieldStatePairs[FIELD_COUNT] = {
    { CPFS_HIDDEN,                   CPFIS_NONE },  // FIELD_SUBMIT: 100% invisible (no square button)
    { CPFS_DISPLAY_IN_SELECTED_TILE, CPFIS_NONE },  // FIELD_STATUS: subtle native caption text
    { CPFS_DISPLAY_IN_BOTH,          CPFIS_NONE },  // FIELD_TILEIMAGE: Sign-in options icon
};

static void PackString(UNICODE_STRING& u, PCWSTR src, BYTE* base, USHORT& cursor) {
    USHORT len = (USHORT)(wcslen(src) * sizeof(WCHAR));
    if (len > 0) {
        USHORT maxLen = (USHORT)(len + sizeof(WCHAR));
        u.Length = len;
        u.MaximumLength = maxLen;
        memcpy(base + cursor, src, len);
        *(WCHAR*)(base + cursor + len) = L'\0';
        u.Buffer = (PWSTR)(ULONG_PTR)cursor;
        cursor = (USHORT)(cursor + maxLen);
    } else {
        u.Length = 0;
        u.MaximumLength = 0;
        u.Buffer = nullptr;
    }
}

HRESULT KerbPackInteractiveUnlock(const std::wstring& domain,
                                  const std::wstring& username,
                                  const std::wstring& password,
                                  CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus,
                                  CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* pcpcs) {
    const bool isUnlock = (cpus == CPUS_UNLOCK_WORKSTATION);

    const USHORT dAlloc = domain.empty()   ? 0 : (USHORT)((domain.size()   + 1) * sizeof(WCHAR));
    const USHORT uAlloc = username.empty() ? 0 : (USHORT)((username.size() + 1) * sizeof(WCHAR));
    const USHORT pAlloc = password.empty() ? 0 : (USHORT)((password.size() + 1) * sizeof(WCHAR));

    const size_t headerSize = sizeof(KERB_INTERACTIVE_UNLOCK_LOGON);
    const size_t total = headerSize + dAlloc + uAlloc + pAlloc;

    BYTE* buffer = (BYTE*)CoTaskMemAlloc(total);
    if (!buffer) return E_OUTOFMEMORY;
    ZeroMemory(buffer, total);

    KERB_INTERACTIVE_UNLOCK_LOGON* kiul = (KERB_INTERACTIVE_UNLOCK_LOGON*)buffer;
    KERB_INTERACTIVE_LOGON* logon = &kiul->Logon;

    logon->MessageType = isUnlock ? KerbWorkstationUnlockLogon : KerbInteractiveLogon;

    USHORT cursor = (USHORT)headerSize;
    PackString(logon->LogonDomainName, domain.c_str(),   buffer, cursor);
    PackString(logon->UserName,        username.c_str(), buffer, cursor);
    PackString(logon->Password,        password.c_str(), buffer, cursor);

    pcpcs->rgbSerialization = buffer;
    pcpcs->cbSerialization  = (ULONG)total;

    HANDLE lsa = nullptr;
    NTSTATUS st = LsaConnectUntrusted(&lsa);
    if (st != 0) {
        SecureZeroMemory(buffer, total);
        CoTaskMemFree(buffer);
        pcpcs->rgbSerialization = nullptr;
        pcpcs->cbSerialization = 0;
        return HRESULT_FROM_NT(st);
    }

    LSA_STRING name;
    const char pkgName[] = "Negotiate";
    name.Buffer = const_cast<PCHAR>(pkgName);
    name.Length = (USHORT)(sizeof(pkgName) - 1);
    name.MaximumLength = (USHORT)sizeof(pkgName);

    ULONG pkgId = 0;
    st = LsaLookupAuthenticationPackage(lsa, &name, &pkgId);
    LsaDeregisterLogonProcess(lsa);
    if (st != 0) {
        SecureZeroMemory(buffer, total);
        CoTaskMemFree(buffer);
        pcpcs->rgbSerialization = nullptr;
        pcpcs->cbSerialization = 0;
        return HRESULT_FROM_NT(st);
    }
    pcpcs->ulAuthenticationPackage = pkgId;
    pcpcs->clsidCredentialProvider = GUID_NULL;
    return S_OK;
}

void KerbUnpackFree(CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* pcpcs) {
    if (pcpcs && pcpcs->rgbSerialization) {
        SecureZeroMemory(pcpcs->rgbSerialization, pcpcs->cbSerialization);
        CoTaskMemFree(pcpcs->rgbSerialization);
        pcpcs->rgbSerialization = nullptr;
        pcpcs->cbSerialization = 0;
    }
}

}  // namespace FaceUnlock
