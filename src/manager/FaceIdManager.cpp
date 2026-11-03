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
#include <commctrl.h>
#include <shellapi.h>

#include "FaceIdHud.h"
#include "NativeFaceEngine.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int IDC_BTN_TEST_FACE   = 1001;
constexpr int IDC_BTN_ENROLL_CAM  = 1002;
constexpr int IDC_BTN_TOGGLE_LOCK = 1004;
constexpr int IDC_EDIT_USER       = 1010;
constexpr int IDC_EDIT_DOM        = 1011;
constexpr int IDC_EDIT_PASS       = 1012;
constexpr int IDC_CHK_SHOW_PASS   = 1013;
constexpr int IDC_BTN_SAVE_PASS   = 1014;

constexpr UINT WM_APP_UPDATE_STATUS = WM_APP + 10;
constexpr UINT WM_APP_ACTION_DONE   = WM_APP + 11;

constexpr wchar_t kProviderRegKey[] =
    L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Authentication\\Credential Providers\\{F8A0B4D9-3C7F-4B0A-9E21-8C1B1E2B7C10}";
constexpr wchar_t kDllPath[] =
    L"C:\\ProgramData\\FaceUnlock\\bin\\FaceCredentialProvider.dll";

struct AppState {
    HWND hwndMain = nullptr;
    HWND lblSummary = nullptr;
    HWND lblResult = nullptr;
    HWND editDomain = nullptr;
    HWND editUser = nullptr;
    HWND editPass = nullptr;
    HWND chkShowPass = nullptr;
    HWND btnTest = nullptr;
    HWND btnEnroll = nullptr;
    HWND btnSavePass = nullptr;
    HWND btnToggleLock = nullptr;
    HFONT fontTitle = nullptr;
    HFONT fontBody = nullptr;
    HFONT fontBold = nullptr;
    bool busy = false;
};

AppState g_app{};

bool IsAdminElevated() {
    BOOL isAdmin = FALSE;
    PSID adminGroup = nullptr;
    SID_IDENTIFIER_AUTHORITY ntAuthority = SECURITY_NT_AUTHORITY;
    if (AllocateAndInitializeSid(&ntAuthority, 2, SECURITY_BUILTIN_DOMAIN_RID,
                                 DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0, 0, 0, &adminGroup)) {
        CheckTokenMembership(nullptr, adminGroup, &isAdmin);
        FreeSid(adminGroup);
    }
    return isAdmin != FALSE;
}

bool IsProviderRegistered() {
    HKEY hKey = nullptr;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, kProviderRegKey, 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegCloseKey(hKey);
        return true;
    }
    return false;
}

std::wstring DetectMsaEmail() {
    HKEY hKey = nullptr;
    const wchar_t* sub = L"Software\\Microsoft\\IdentityCRL\\UserExtendedProperties";
    if (RegOpenKeyExW(HKEY_CURRENT_USER, sub, 0, KEY_READ, &hKey) != ERROR_SUCCESS) {
        return L"";
    }
    wchar_t nameBuf[256]{};
    DWORD nameLen = 256;
    std::wstring result;
    for (DWORD idx = 0; ; ++idx) {
        nameLen = 256;
        if (RegEnumKeyExW(hKey, idx, nameBuf, &nameLen, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) {
            break;
        }
        std::wstring s(nameBuf, nameLen);
        if (s.find(L'@') != std::wstring::npos) {
            result = s;
            break;
        }
    }
    RegCloseKey(hKey);
    return result;
}

bool VerifyWindowsLsaPassword(const std::wstring& domain,
                              const std::wstring& user,
                              const std::wstring& pass,
                              DWORD& outErr) {
    HANDLE hTok = nullptr;
    const wchar_t* domPtr = domain.empty() ? L"" : domain.c_str();
    if (LogonUserW(user.c_str(), domPtr, pass.c_str(),
                   LOGON32_LOGON_UNLOCK, LOGON32_PROVIDER_WINNT50, &hTok)) {
        CloseHandle(hTok);
        outErr = 0;
        return true;
    }
    if (LogonUserW(user.c_str(), domPtr, pass.c_str(),
                   LOGON32_LOGON_INTERACTIVE, LOGON32_PROVIDER_DEFAULT, &hTok)) {
        CloseHandle(hTok);
        outErr = 0;
        return true;
    }
    outErr = GetLastError();
    return false;
}

void SetControlsEnabled(bool en) {
    g_app.busy = !en;
    EnableWindow(g_app.btnTest, en ? TRUE : FALSE);
    EnableWindow(g_app.btnEnroll, en ? TRUE : FALSE);
    EnableWindow(g_app.btnSavePass, en ? TRUE : FALSE);
    EnableWindow(g_app.btnToggleLock, en ? TRUE : FALSE);
}

void RefreshDashboardText() {
    auto info = FaceUnlock::NativeQueryVaultInfo();
    bool reg = IsProviderRegistered();

    wchar_t buf[1024]{};
    std::swprintf(
        buf, 1024,
        L"• Kiến trúc: 100%% C++20 Zero-Daemon (0 MB RAM chạy nền — chỉ kích hoạt khi mở khóa)\r\n"
        L"• Màn hình khóa Windows 11: %ls\r\n"
        L"• Dữ liệu khuôn mặt (SFace + MiniFASNet ONNX): %ls (%u mẫu, ngưỡng %.2f)\r\n"
        L"• Kho mật khẩu bảo mật DPAPI: %ls",
        reg ? L"ĐANG BẬT (Tự quét kèm Dynamic Island 120–165 FPS ProMotion)" : L"ĐANG TẮT",
        info.hasEmbeddings ? L"SẴN SÀNG" : L"CHƯA ĐĂNG KÝ",
        info.enrolledCount,
        info.threshold,
        info.hasCredentials
            ? (info.domain + L"\\" + info.username).c_str()
            : L"CHƯA CẤU HÌNH");

    SetWindowTextW(g_app.lblSummary, buf);
    SetWindowTextW(g_app.btnToggleLock,
                   reg ? L"⏸ Tạm tắt Face ID trên Màn hình khóa"
                       : L"▶ Bật Face ID trên Màn hình khóa");
}

void RunLiveVerifyTestAsync() {
    SetControlsEnabled(false);
    SetWindowTextW(g_app.lblResult, L"⏳ Đang bật camera & quét Face ID (Dynamic Island 120+ FPS)...");

    std::thread([]() {
        FaceUnlock::FaceIdHud hud;
        hud.StartScanning();
        auto stats = FaceUnlock::NativeVerifyLiveCamera(6);
        hud.FinishWithResult(stats.matched);

        auto* msg = new std::wstring();
        wchar_t line[512]{};
        if (stats.matched) {
            std::swprintf(line, 512,
                          L"✅ KHỚP KHUÔN MẶT THÀNH CÔNG!  |  Độ tương đồng: %.1f%%  |  Liveness: Thật (%.2f)  |  Tổng thời gian: %.0f ms (Camera đã tắt)",
                          stats.bestSimilarity * 100.0f, stats.realScore, stats.elapsedMs);
        } else {
            std::swprintf(line, 512,
                          L"❌ KHÔNG KHỚP (%hs)  |  Độ tương đồng cao nhất: %.1f%%  |  Thời gian: %.0f ms (Camera đã tắt)",
                          stats.error.c_str(), stats.bestSimilarity * 100.0f, stats.elapsedMs);
        }
        *msg = line;
        PostMessageW(g_app.hwndMain, WM_APP_ACTION_DONE, 0, reinterpret_cast<LPARAM>(msg));
    }).detach();
}

void RunWebcamEnrollAsync() {
    SetControlsEnabled(false);
    SetWindowTextW(g_app.lblResult, L"📷 Hãy nhìn thẳng và xoay nhẹ mặt trước camera (0/15 mẫu)...");

    std::thread([]() {
        uint32_t count = 0;
        std::string err;

        auto cb = [](uint32_t captured, uint32_t target, const uint8_t*, int, int,
                     const FaceUnlock::FaceBox5* face, void*) {
            auto* statusText = new std::wstring();
            wchar_t tmp[256]{};
            std::swprintf(tmp, 256,
                          L"📷 Đang thu thập mẫu khuôn mặt C++20: %u/%u mẫu %ls",
                          captured, target,
                          face ? L"(Đã thấy khuôn mặt - giữ nguyên / xoay rất nhẹ)" : L"(Chưa thấy khuôn mặt...)");
            *statusText = tmp;
            PostMessageW(g_app.hwndMain, WM_APP_UPDATE_STATUS, 0, reinterpret_cast<LPARAM>(statusText));
        };

        bool ok = FaceUnlock::NativeEnrollFromWebcam(15, 0.45f, true, cb, nullptr, count, err);
        auto* doneMsg = new std::wstring();
        wchar_t line[384]{};
        if (ok) {
            std::swprintf(line, 384,
                          L"✅ Đã đăng ký mới %u mẫu khuôn mặt vào embeddings.bin thành công! Camera đã tắt hoàn toàn.",
                          count);
        } else {
            std::swprintf(line, 384, L"❌ Đăng ký khuôn mặt thất bại: %hs (Camera đã tắt).", err.c_str());
        }
        *doneMsg = line;
        PostMessageW(g_app.hwndMain, WM_APP_ACTION_DONE, 1, reinterpret_cast<LPARAM>(doneMsg));
    }).detach();
}

void OnSavePasswordClicked(HWND hwnd) {
    wchar_t domBuf[256]{}, userBuf[256]{}, passBuf[256]{};
    GetWindowTextW(g_app.editDomain, domBuf, 256);
    GetWindowTextW(g_app.editUser, userBuf, 256);
    GetWindowTextW(g_app.editPass, passBuf, 256);

    std::wstring dom(domBuf), usr(userBuf), pw(passBuf);
    SecureZeroMemory(passBuf, sizeof(passBuf));

    if (usr.empty() || pw.empty()) {
        MessageBoxW(hwnd, L"Vui lòng nhập đầy đủ Tài khoản và Mật khẩu (không nhập mã PIN 6 số).",
                    L"Thiếu thông tin", MB_ICONWARNING | MB_OK);
        if (!pw.empty()) SecureZeroMemory(pw.data(), pw.size() * sizeof(wchar_t));
        return;
    }

    SetWindowTextW(g_app.lblResult, L"⏳ Đang xác thực thử mật khẩu trực tiếp với Windows LSA...");
    UpdateWindow(hwnd);

    DWORD winErr = 0;
    bool verified = VerifyWindowsLsaPassword(dom, usr, pw, winErr);
    if (!verified && dom == L"MicrosoftAccount") {
        // Also try empty UPN domain
        if (VerifyWindowsLsaPassword(L"", usr, pw, winErr)) {
            dom = L"";
            verified = true;
        }
    }

    if (!verified) {
        wchar_t msg[640]{};
        std::swprintf(msg, 640,
                      L"Windows LSA từ chối mật khẩu (Mã lỗi %lu).\n\n"
                      L"• Lưu ý nhập MẬT KHẨU chữ+số của tài khoản Microsoft/Windows, KHÔNG nhập mã PIN 6 số.\n"
                      L"• Bạn có muốn lưu bắt buộc vào Vault không?",
                      winErr);
        int ans = MessageBoxW(hwnd, msg, L"Xác thực LSA không khớp", MB_ICONWARNING | MB_YESNO);
        if (ans != IDYES) {
            SecureZeroMemory(pw.data(), pw.size() * sizeof(wchar_t));
            SetWindowTextW(g_app.lblResult, L"❌ Đã hủy lưu mật khẩu do không khớp LSA.");
            return;
        }
    }

    std::string err;
    bool saved = FaceUnlock::NativeSaveVaultCredentials(dom, usr, pw, err);
    SecureZeroMemory(pw.data(), pw.size() * sizeof(wchar_t));
    SetWindowTextW(g_app.editPass, L"");

    if (saved) {
        RefreshDashboardText();
        SetWindowTextW(g_app.lblResult,
                       verified
                           ? L"✅ Xác thực Windows LSA đúng 100% & đã mã hóa lưu vào DPAPI Vault!"
                           : L"✅ Đã mã hóa lưu vào DPAPI Vault.");
        MessageBoxW(hwnd,
                    L"Đã mã hóa DPAPI và lưu mật khẩu thành công!\nBây giờ bạn có thể bấm Win + L để mở khóa khuôn mặt ngay.",
                    L"Thành công", MB_ICONINFORMATION | MB_OK);
    } else {
        wchar_t emsg[256]{};
        std::swprintf(emsg, 256, L"Lỗi ghi DPAPI Vault (%hs). Hãy chạy bằng quyền Administrator.", err.c_str());
        MessageBoxW(hwnd, emsg, L"Lỗi", MB_ICONERROR | MB_OK);
    }
}

void OnToggleProviderClicked(HWND hwnd) {
    HMODULE hDll = LoadLibraryExW(kDllPath, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!hDll) {
        MessageBoxW(hwnd, L"Không tìm thấy FaceCredentialProvider.dll!", L"Lỗi", MB_ICONERROR | MB_OK);
        return;
    }
    using RegFn = HRESULT(STDAPICALLTYPE*)();
    bool currentlyReg = IsProviderRegistered();
    auto fn = reinterpret_cast<RegFn>(GetProcAddress(
        hDll, currentlyReg ? "DllUnregisterServer" : "DllRegisterServer"));
    HRESULT hr = fn ? fn() : E_FAIL;
    FreeLibrary(hDll);

    if (SUCCEEDED(hr)) {
        RefreshDashboardText();
        SetWindowTextW(g_app.lblResult,
                       currentlyReg
                           ? L"⏸ Đã tạm tắt Face ID trên màn hình khóa."
                           : L"✅ Đã bật Face ID trên màn hình khóa Windows 11!");
    } else {
        MessageBoxW(hwnd, L"Không thể cập nhật Registry (cần quyền Administrator).", L"Lỗi", MB_ICONERROR | MB_OK);
    }
}

LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            g_app.fontTitle = CreateFontW(-22, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                          DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                          CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
            g_app.fontBold  = CreateFontW(-15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
                                          DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                          CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
            g_app.fontBody  = CreateFontW(-14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                          DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                          CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

            HWND hHeader = CreateWindowExW(0, L"STATIC",
                L"⚡ Windows 11 Face ID — Bản 100% C++20 Zero-Daemon",
                WS_CHILD | WS_VISIBLE, 22, 18, 600, 32, hwnd, nullptr, nullptr, nullptr);
            SendMessageW(hHeader, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontTitle), TRUE);

            g_app.lblSummary = CreateWindowExW(WS_EX_CLIENTEDGE, L"STATIC", L"",
                WS_CHILD | WS_VISIBLE | SS_LEFT, 22, 58, 600, 96, hwnd, nullptr, nullptr, nullptr);
            SendMessageW(g_app.lblSummary, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);

            g_app.btnTest = CreateWindowExW(0, L"BUTTON",
                L"📸 Quét thử Face ID ngay (Dynamic Island 120+ FPS)",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 22, 168, 292, 42,
                hwnd, reinterpret_cast<HMENU>(IDC_BTN_TEST_FACE), nullptr, nullptr);
            SendMessageW(g_app.btnTest, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBold), TRUE);

            g_app.btnEnroll = CreateWindowExW(0, L"BUTTON",
                L"🧑 Quét & Đăng ký lại 15 mẫu mặt (C++)",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 330, 168, 292, 42,
                hwnd, reinterpret_cast<HMENU>(IDC_BTN_ENROLL_CAM), nullptr, nullptr);
            SendMessageW(g_app.btnEnroll, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBold), TRUE);

            g_app.btnToggleLock = CreateWindowExW(0, L"BUTTON", L"",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 22, 218, 600, 36,
                hwnd, reinterpret_cast<HMENU>(IDC_BTN_TOGGLE_LOCK), nullptr, nullptr);
            SendMessageW(g_app.btnToggleLock, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);

            // Credentials section
            HWND grpCred = CreateWindowExW(0, L"BUTTON",
                L"🔑 Cập nhật Mật khẩu Mở khóa (Kiểm tra LSA & Mã hóa DPAPI)",
                WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 22, 268, 600, 152, hwnd, nullptr, nullptr, nullptr);
            SendMessageW(grpCred, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBold), TRUE);

            auto info = FaceUnlock::NativeQueryVaultInfo();
            std::wstring msa = DetectMsaEmail();
            wchar_t localUserBuf[256]{};
            DWORD localUserLen = 256;
            std::wstring fallbackUser = L"User";
            if (GetUserNameW(localUserBuf, &localUserLen) && localUserBuf[0] != L'\0') {
                fallbackUser = localUserBuf;
            }
            std::wstring initDom = info.hasCredentials ? info.domain : (!msa.empty() ? L"MicrosoftAccount" : L".");
            std::wstring initUsr = info.hasCredentials ? info.username : (!msa.empty() ? msa : fallbackUser);

            HWND lDom = CreateWindowExW(0, L"STATIC", L"Domain:", WS_CHILD | WS_VISIBLE,
                                        38, 298, 70, 24, hwnd, nullptr, nullptr, nullptr);
            SendMessageW(lDom, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);
            g_app.editDomain = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", initDom.c_str(),
                                               WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
                                               110, 295, 165, 26, hwnd, reinterpret_cast<HMENU>(IDC_EDIT_DOM), nullptr, nullptr);
            SendMessageW(g_app.editDomain, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);

            HWND lUsr = CreateWindowExW(0, L"STATIC", L"Tài khoản:", WS_CHILD | WS_VISIBLE,
                                        290, 298, 75, 24, hwnd, nullptr, nullptr, nullptr);
            SendMessageW(lUsr, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);
            g_app.editUser = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", initUsr.c_str(),
                                             WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
                                             368, 295, 238, 26, hwnd, reinterpret_cast<HMENU>(IDC_EDIT_USER), nullptr, nullptr);
            SendMessageW(g_app.editUser, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);

            HWND lPw = CreateWindowExW(0, L"STATIC", L"Mật khẩu:", WS_CHILD | WS_VISIBLE,
                                       38, 336, 70, 24, hwnd, nullptr, nullptr, nullptr);
            SendMessageW(lPw, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);
            g_app.editPass = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                                             WS_CHILD | WS_VISIBLE | ES_PASSWORD | ES_AUTOHSCROLL,
                                             110, 333, 300, 26, hwnd, reinterpret_cast<HMENU>(IDC_EDIT_PASS), nullptr, nullptr);
            SendMessageW(g_app.editPass, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);

            g_app.btnSavePass = CreateWindowExW(0, L"BUTTON", L"✅ Kiểm tra & Lưu",
                                                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                                422, 331, 184, 30,
                                                hwnd, reinterpret_cast<HMENU>(IDC_BTN_SAVE_PASS), nullptr, nullptr);
            SendMessageW(g_app.btnSavePass, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBold), TRUE);

            g_app.chkShowPass = CreateWindowExW(0, L"BUTTON", L"Hiện mật khẩu (không nhập mã PIN 6 số)",
                                                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                                110, 366, 320, 24,
                                                hwnd, reinterpret_cast<HMENU>(IDC_CHK_SHOW_PASS), nullptr, nullptr);
            SendMessageW(g_app.chkShowPass, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);

            g_app.lblResult = CreateWindowExW(0, L"STATIC",
                L"Sẵn sàng — Không có tiến trình ngầm nào chạy tốn RAM. Đóng cửa sổ này Face ID trên màn hình khóa vẫn hoạt động 100%.",
                WS_CHILD | WS_VISIBLE | SS_LEFT, 22, 432, 600, 44, hwnd, nullptr, nullptr, nullptr);
            SendMessageW(g_app.lblResult, WM_SETFONT, reinterpret_cast<WPARAM>(g_app.fontBody), TRUE);

            RefreshDashboardText();
            return 0;
        }

        case WM_COMMAND: {
            int id = LOWORD(wParam);
            if (id == IDC_BTN_TEST_FACE && !g_app.busy) {
                RunLiveVerifyTestAsync();
            } else if (id == IDC_BTN_ENROLL_CAM && !g_app.busy) {
                RunWebcamEnrollAsync();
            } else if (id == IDC_BTN_SAVE_PASS && !g_app.busy) {
                OnSavePasswordClicked(hwnd);
            } else if (id == IDC_BTN_TOGGLE_LOCK && !g_app.busy) {
                OnToggleProviderClicked(hwnd);
            } else if (id == IDC_CHK_SHOW_PASS) {
                bool show = (SendMessageW(g_app.chkShowPass, BM_GETCHECK, 0, 0) == BST_CHECKED);
                SendMessageW(g_app.editPass, EM_SETPASSWORDCHAR, show ? 0 : L'●', 0);
                InvalidateRect(g_app.editPass, nullptr, TRUE);
            }
            return 0;
        }

        case WM_APP_UPDATE_STATUS: {
            auto* s = reinterpret_cast<std::wstring*>(lParam);
            if (s) {
                SetWindowTextW(g_app.lblResult, s->c_str());
                delete s;
            }
            return 0;
        }

        case WM_APP_ACTION_DONE: {
            auto* s = reinterpret_cast<std::wstring*>(lParam);
            if (s) {
                SetWindowTextW(g_app.lblResult, s->c_str());
                delete s;
            }
            SetControlsEnabled(true);
            RefreshDashboardText();
            return 0;
        }

        case WM_CTLCOLORSTATIC: {
            HDC hdc = reinterpret_cast<HDC>(wParam);
            SetBkMode(hdc, TRANSPARENT);
            return reinterpret_cast<LRESULT>(GetSysColorBrush(COLOR_WINDOW));
        }

        case WM_DESTROY: {
            if (g_app.fontTitle) DeleteObject(g_app.fontTitle);
            if (g_app.fontBold)  DeleteObject(g_app.fontBold);
            if (g_app.fontBody)  DeleteObject(g_app.fontBody);
            PostQuitMessage(0);
            return 0;
        }
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR lpCmdLine, int nCmdShow) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    // Optional CLI modes: --test-live or --enroll-dir <path>
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argv && argc >= 2) {
        std::wstring arg1 = argv[1];
        if (arg1 == L"--test-live") {
            LocalFree(argv);
            auto st = FaceUnlock::NativeVerifyLiveCamera(6);
            std::printf("matched=%d face=%d real=%d sim=%.4f realScore=%.4f elapsed=%.1fms err=%s\n",
                        st.matched ? 1 : 0, st.faceDetected ? 1 : 0, st.isReal ? 1 : 0,
                        st.bestSimilarity, st.realScore, st.elapsedMs, st.error.c_str());
            return st.matched ? 0 : 1;
        }
        if (arg1 == L"--enroll-dir" && argc >= 3) {
            std::wstring dir = argv[2];
            LocalFree(argv);
            uint32_t cnt = 0;
            std::string err;
            bool ok = FaceUnlock::NativeBuildEnrollmentFromDir(dir, cnt, 0.45f, true, err);
            std::printf("enroll_ok=%d count=%u err=%s\n", ok ? 1 : 0, cnt, err.c_str());
            return ok ? 0 : 1;
        }
    }
    if (argv) LocalFree(argv);
    (void)lpCmdLine;

    // Self-elevate to Administrator so Vault & Registry actions always work seamlessly
    if (!IsAdminElevated()) {
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        SHELLEXECUTEINFOW sei{};
        sei.cbSize = sizeof(sei);
        sei.lpVerb = L"runas";
        sei.lpFile = exePath;
        sei.nShow  = SW_SHOWNORMAL;
        if (ShellExecuteExW(&sei)) {
            return 0;
        }
    }

    INITCOMMONCONTROLSEX icc{ sizeof(icc), ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&icc);

    const wchar_t* clsName = L"FaceIdNativeManagerCpp20";
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = MainWndProc;
    wc.hInstance     = hInst;
    wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon         = LoadIconW(nullptr, IDI_SHIELD);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    wc.lpszClassName = clsName;
    RegisterClassExW(&wc);

    int w = 660, h = 530;
    int sx = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
    int sy = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;

    HWND hwnd = CreateWindowExW(
        0, clsName,
        L"Face ID Control Center — 100% C++20 Zero-Daemon",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        sx, sy, w, h,
        nullptr, nullptr, hInst, nullptr);

    if (!hwnd) return 1;
    g_app.hwndMain = hwnd;

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return static_cast<int>(msg.wParam);
}
