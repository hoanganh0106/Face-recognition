# ⚡ Windows 11 Face ID — 100% Native C++20 Zero-Daemon + 120–165Hz iOS Dynamic Island

> Mở khóa khuôn mặt siêu tốc cho **mọi laptop / PC chạy Windows 11 chỉ có webcam thường (không cần camera hồng ngoại IR Windows Hello)** — viết thuần **100% C++20 (`-O3 -mavx2 -mfma`)**, **tốn `0.0 MB` RAM chạy ngầm**, tự động tắt cứng nguồn webcam ngay khoảnh khắc mở khóa xong, đi kèm hiệu ứng **Apple iOS Dynamic Island ProMotion 120–165 FPS** trực tiếp trên Màn hình khóa Windows 11.

---

## ✨ Điểm nổi bật (Key Highlights)

- **🚀 Zero-Daemon (`0 MB` RAM chạy nền)**: Không dùng Python, không nuôi service chạy ngầm 24/7, không tốn pin. Toàn bộ nhân nhận diện chỉ được `LogonUI.exe` của Windows gọi dậy đúng ~`1 giây` khi bạn đánh thức màn hình khóa (`Win + L`, mở nắp máy, hoặc bật nguồn khởi động máy).
- **🔒 Tắt cứng đèn Webcam tức thì (`100% Camera OFF` khi dùng máy)**: Đọc luồng ảnh trực tiếp qua Win32 MediaFoundation (`IMFSourceReader` + `IMF2DBuffer::Lock2D`) và gọi `IMFMediaSource::Shutdown()` tắt đèn LED phần cứng ngay khoảnh khắc khớp khuôn mặt (~`0.8–1.0s`).
- **🏝️ Hiệu ứng iOS Dynamic Island ProMotion 120–165 FPS**: Hiển thị trực tiếp trên màn hình khóa Windows (`WinSta0\Winlogon`) với nội suy lò xo vật lý sub-pixel, đồng hồ `QueryPerformanceCounter` dưới `1 µs`, và bộ điều nhịp **NT Kernel High-Resolution Timer** tự động khóa theo tần số quét thật của màn hình (`120Hz` / `144Hz` / `165Hz` / `240Hz`).
- **🛡️ Chống giả mạo 2 lớp (Dual-Scale Liveness Anti-Spoofing)**: Kết hợp 4 mô hình ONNX tối ưu hóa đồ thị (`ORT_ENABLE_ALL`):
  - **YuNet (`yunet_opt.onnx`)**: Phát hiện khuôn mặt + 5 điểm mốc (mắt, mũi, khóe miệng).
  - **Umeyama 5-Point Similarity Alignment**: Căn chỉnh hình học khuôn mặt chuẩn xác bằng giải tích đóng C++20.
  - **Dual MiniFASNetV2 + MiniFASNetV1SE (`fas1_opt.onnx`, `fas2_opt.onnx`)**: Chống dùng ảnh chụp / màn hình điện thoại đưa lên trước camera.
  - **SFace 128-D (`sface_opt.onnx`)**: Đối sánh khoảng cách Cosine độ chính xác cao.
- **🔐 Két mật khẩu phần cứng DPAPI (`CRYPTPROTECT_LOCAL_MACHINE`)**: Xác thực mật khẩu trực tiếp với Windows LSA (`LogonUserW`), hỗ trợ trọn vẹn cả **Tài khoản Microsoft (`MicrosoftAccount\email@...`)** đang dùng kèm mã PIN lẫn **Tài khoản Local (`.`)**, khóa quyền truy cập NTFS thư mục `vault\` chỉ dành riêng cho `NT AUTHORITY\SYSTEM` và `Administrators`.

---

## 📦 Yêu cầu hệ thống

- Hệ điều hành: **Windows 10 / Windows 11 (64-bit x64)**
- Phần cứng: Bất kỳ webcam tích hợp trên laptop hoặc webcam rời USB (không cần camera IR)
- Quyền hạn: Quyền **Administrator** trong lần cài đặt đầu tiên (không cần cài Python, không cần cài Visual Studio hay thư viện ngoài — bản biên dịch sẵn đã đi kèm đầy đủ trong repo).

---

## 🚀 Hướng dẫn cài đặt nhanh trong 60 giây (Quick Start)

### Bước 1: Tải bộ cài về máy
```powershell
git clone https://github.com/hoanganh0106/Face-recognition.git
cd Face-recognition
```
*(Hoặc bấm **Code ➔ Download ZIP** trên GitHub rồi giải nén ra thư mục bất kỳ).*

### Bước 2: Chạy cài đặt tự động 1 chạm
Nhấp đúp chuột vào tệp **`Install.cmd`** (bấm **Yes** khi Windows hỏi quyền Administrator), hoặc chạy trong PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```
Trình cài đặt sẽ tự động:
1. Tạo vùng lưu trữ hệ thống tại `C:\ProgramData\FaceUnlock\` (độc lập hoàn toàn, khởi động lại máy vẫn hoạt động vĩnh viễn và bạn có thể dời/xóa thư mục tải về tùy ý sau khi cài).
2. Phân quyền bảo mật NTFS cho két `C:\ProgramData\FaceUnlock\vault`.
3. Mở khóa cho phép dùng song song Mật khẩu Microsoft Account cùng Windows Hello PIN (`DevicePasswordLessBuildVersion = 0`).
4. Đăng ký `FaceCredentialProvider.dll` vào Màn hình khóa Windows 11 và mở ngay ứng dụng **Face ID Control Center**.

### Bước 3: Đăng ký khuôn mặt & Lưu mật khẩu mở khóa
Ngay trong cửa sổ **Face ID Control Center** vừa hiện lên (hoặc mở lại bất cứ lúc nào qua `FaceIdManager.exe` / `Chay_UI_Test.cmd`):

1. **Quét mẫu khuôn mặt**:
   - Bấm nút **🧑 Quét & Đăng ký lại 15 mẫu mặt (C++)**.
   - Nhìn thẳng vào webcam và nghiêng đầu rất nhẹ trái/phải trong khoảng `5–7 giây` cho tới khi thu đủ `15/15` mẫu. Camera sẽ tự động tắt ngay sau khi lưu xong `embeddings.bin`.
2. **Cấu hình mật khẩu Windows**:
   - **Nếu máy bạn đăng nhập bằng Tài khoản Microsoft**:
     - Ô `Domain`: để `MicrosoftAccount`
     - Ô `Tài khoản`: nhập email Microsoft của bạn (ứng dụng tự động điền sẵn email đang đăng nhập trên máy)
     - Ô `Mật khẩu`: nhập **Mật khẩu tài khoản Microsoft** (chữ + số, **lưu ý KHÔNG nhập mã PIN 6 số**).
   - **Nếu máy bạn dùng Tài khoản Nội bộ (Local Account)**:
     - Ô `Domain`: để dấu chấm `.`
     - Ô `Tài khoản`: tên user Windows của bạn (ứng dụng tự động điền sẵn)
     - Ô `Mật khẩu`: mật khẩu đăng nhập máy.
   - Bấm **✅ Kiểm tra & Lưu** ➔ Ứng dụng sẽ kiểm tra trực tiếp với nhân Windows LSA để đảm bảo mật khẩu đúng 100% trước khi mã hóa vào két DPAPI.
3. **Kiểm tra thử & Sử dụng**:
   - Bấm **📸 Quét thử Face ID ngay (Dynamic Island 120+ FPS)** để xem hiệu ứng viên thuốc Dynamic Island bung ra trên đỉnh màn hình và kiểm tra độ nhạy.
   - Đóng cửa sổ lại (không cần treo ngầm!), nhấn tổ hợp phím **`Win + L`** để khóa máy ➔ Nhấp chuột hoặc nhấn phím bất kỳ đánh thức màn hình khóa, Face ID sẽ tự động bung Dynamic Island và mở khóa thẳng vào Desktop trong ~`1 giây`!

---

## 🛠️ Cấu trúc thư mục dự án

```text
Face-recognition/
├── Install.cmd                        # Trình cài đặt tự động 1 chạm (Auto-Admin)
├── Uninstall.cmd                      # Gỡ cài đặt sạch sẽ khỏi Màn hình khóa & xóa dữ liệu
├── Chay_UI_Test.cmd                   # Mở nhanh bảng điều khiển Face ID Control Center
├── FaceIdManager.exe                  # Ứng dụng quản lý C++20 Win32 (Biên dịch sẵn x64 AVX2)
├── CMakeLists.txt                     # Cấu hình build CMake chuẩn (MSVC / Clang)
├── dist/
│   ├── FaceCredentialProvider.dll     # Nhân Windows Credential Provider + HUD 120-165Hz
│   ├── onnxruntime.dll                # Nhân suy luận ONNX Runtime C-API x64
│   └── onnxruntime_providers_shared.dll
├── models/                            # Bộ 4 mô hình ONNX đã tối ưu hóa hằng số (ORT_ENABLE_ALL)
│   ├── yunet_opt.onnx                 # Phát hiện khuôn mặt + 5 điểm mốc
│   ├── sface_opt.onnx                 # Trích xuất vector đặc trưng 128 chiều SFace
│   ├── fas1_opt.onnx                  # Chống giả mạo khuôn mặt MiniFASNetV2 (Scale 2.7x)
│   └── fas2_opt.onnx                  # Chống giả mạo khuôn mặt MiniFASNetV1SE (Scale 4.0x)
├── src/
│   ├── credential_provider/           # Mã nguồn C++20 Credential Provider, Dynamic Island HUD & Engine
│   │   ├── FaceCredentialProvider.*   # Giao tiếp COM ICredentialProvider với Windows LogonUI
│   │   ├── FaceCredential.*           # Luồng Auto-Logon khi đánh thức màn hình khóa
│   │   ├── FaceIdHud.*                # Đồ họa ProMotion 120-165 FPS Dynamic Island (GDI+ Layered Window)
│   │   ├── NativeFaceEngine.*         # MediaFoundation Webcam + Umeyama + ONNX Runtime + DPAPI Vault
│   │   ├── helpers.*                  # Đóng gói gói tin xác thực Kerberos/Negotiate/CloudAP cho LSA
│   │   ├── ClassFactory.cpp           # COM ClassFactory
│   │   └── dll.cpp                    # Điểm vào DLL & Đăng ký HKLM Registry
│   └── manager/
│       └── FaceIdManager.cpp          # Mã nguồn giao diện điều khiển & CLI đăng ký/kiểm thử
├── third_party/
│   └── onnxruntime_c_api.h            # Header C-API chính thức của Microsoft ONNX Runtime
└── scripts/
    ├── install.ps1                    # Kịch bản triển khai hệ thống
    ├── uninstall.ps1                  # Kịch bản gỡ cài đặt an toàn
    └── build.ps1                      # Kịch bản tự biên dịch lại từ mã nguồn C++20
```

---

## ⚙️ Tự biên dịch từ mã nguồn C++20 (Build From Source)

Nếu bạn muốn tự sửa đổi mã nguồn và biên dịch lại `FaceCredentialProvider.dll` cùng `FaceIdManager.exe`:

### Cách 1: Dùng LLVM-MinGW (`clang++`) hoặc CMake tự động
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

### Cách 2: Dùng CMake + Visual Studio 2022 (MSVC)
```powershell
cmake -S . -B build_cmake -A x64
cmake --build build_cmake --config Release --parallel
```
Sau khi build xong, chạy lại `.\Install.cmd` (khi máy đang mở khóa ở Desktop) để cập nhật tệp `.dll` mới vào `C:\ProgramData\FaceUnlock\bin\`.

---

## ❓ Câu hỏi thường gặp & Xử lý sự cố (FAQ & Troubleshooting)

1. **Nếu phòng tối đen hoặc che camera thì có bị kẹt ngoài màn hình khóa không?**
   - **Không bao giờ.** Nếu sau ~`1 giây` không thấy mặt hoặc không khớp, Dynamic Island chỉ **nháy đỏ nhẹ kèm rung ngang kiểu iOS** rồi thu gọn vào cạnh trên màn hình (hoàn toàn không bật hộp thoại báo lỗi rườm rà của Windows). Bạn chỉ cần gõ nhẹ **phím `Space` (Phím cách)**, **`Enter`**, phím **`F`**, **nhấp đúp `Ctrl`**, hoặc bấm chữ **`Quét lại Face ID`** để quét lại ngay lập tức — hoặc chọn **Sign-in options** bên dưới để nhập **PIN / Mật khẩu** bình thường.
2. **Tại sao nhập mã PIN 6 số vào ô Mật khẩu lại báo lỗi LSA?**
   - Mã PIN của Windows Hello được gắn cứng với chip TPM nội bộ của provider Microsoft PIN, các Credential Provider bên thứ ba bắt buộc xác thực qua gói `Negotiate` / `CloudAP` bằng **Mật khẩu tài khoản Microsoft (hoặc mật khẩu Local)**. Khi bạn lưu mật khẩu Microsoft vào ứng dụng, mã PIN 6 số cũ của bạn trên màn hình khóa vẫn giữ nguyên không bị mất.
3. **Khi đổi mật khẩu tài khoản Microsoft / Windows thì làm gì?**
   - Mở `FaceIdManager.exe` (hoặc `C:\ProgramData\FaceUnlock\bin\FaceIdManager.exe`), nhập mật khẩu mới và bấm **✅ Kiểm tra & Lưu**.
4. **Làm sao để tạm tắt hoặc gỡ bỏ hoàn toàn khỏi máy?**
   - **Tạm tắt 1 chạm**: Mở `FaceIdManager.exe` ➔ bấm nút **⏸ Tạm tắt Face ID trên Màn hình khóa**.
   - **Gỡ bỏ hoàn toàn**: Nhấp đúp vào **`Uninstall.cmd`** (tự động hủy đăng ký trên màn hình khóa và xóa sạch toàn bộ dữ liệu khuôn mặt + két mật khẩu tại `C:\ProgramData\FaceUnlock`).

---

## 🙏 Credits & Attribution

- Mô hình nhận diện & phát hiện khuôn mặt: **YuNet** & **SFace** thuộc [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (Apache-2.0).
- Mô hình chống giả mạo (Liveness Detection): **MiniFASNetV2 / MiniFASNetV1SE** thuộc [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (Apache-2.0).
- Tham khảo giao thức Windows Credential Provider ban đầu từ [caochitam/windows-face-unlock](https://github.com/caochitam/windows-face-unlock), tái kiến trúc mới toàn diện sang **100% Native C++20 Zero-Daemon + MediaFoundation + ProMotion 120–165Hz Dynamic Island HUD**.
