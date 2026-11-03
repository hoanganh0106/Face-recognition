#pragma once
#include <windows.h>
#include <credentialprovider.h>
#include <ntsecapi.h>
#include <string>

namespace FaceUnlock {

struct FIELD_STATE_PAIR {
    CREDENTIAL_PROVIDER_FIELD_STATE cpfs;
    CREDENTIAL_PROVIDER_FIELD_INTERACTIVE_STATE cpfis;
};

// Field indices for our clean, square-free Windows 11 Lock Screen tile UI.
enum FIELD_ID : DWORD {
    FIELD_TILE_IMAGE = 0,
    FIELD_LABEL      = 1,
    FIELD_SUBMIT     = 2,  // Hidden (CPFS_HIDDEN) so Windows draws ZERO ugly square arrow button
    FIELD_STATUS     = 3,
    FIELD_RESCAN     = 4,  // Clean borderless hyperlink ("Quét lại Face ID")
    FIELD_COUNT
};

extern const CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR s_FieldDescriptors[FIELD_COUNT];
extern const FIELD_STATE_PAIR s_FieldStatePairs[FIELD_COUNT];

HRESULT KerbPackInteractiveUnlock(const std::wstring& domain,
                                  const std::wstring& username,
                                  const std::wstring& password,
                                  CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus,
                                  CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* pcpcs);

void KerbUnpackFree(CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* pcpcs);

}  // namespace FaceUnlock
