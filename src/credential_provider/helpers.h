#pragma once
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NTDDI_VERSION
#define NTDDI_VERSION 0x0A000000
#endif
#include <windows.h>
#include <credentialprovider.h>
#include <ntsecapi.h>
#include <string>

namespace FaceUnlock {

struct FIELD_STATE_PAIR {
    CREDENTIAL_PROVIDER_FIELD_STATE cpfs;
    CREDENTIAL_PROVIDER_FIELD_INTERACTIVE_STATE cpfis;
};

// Minimalist V2 User-Bound layout: ZERO custom tile icons, ZERO square buttons.
// Windows 11 keeps the user's native circular profile picture + display name,
// while ONLY the 120–165 FPS Dynamic Island animates at the top of the screen.
enum FIELD_ID : DWORD {
    FIELD_SUBMIT = 0,  // Hidden (CPFS_HIDDEN) — satisfies LogonUI auto-logon with 0 visible UI
    FIELD_STATUS = 1,  // Subtle 1-line hint text under user's native Windows 11 name
    FIELD_TILEIMAGE = 2,  // Icon shown for Face ID in the "Sign-in options" list
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
