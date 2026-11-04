#pragma once
#include <windows.h>
#include <cstdint>
#include <string>
#include <vector>

namespace FaceUnlock {

struct FaceBox5 {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float landmarks[5][2]{};  // right_eye, left_eye, nose, right_mouth, left_mouth
    float score = 0.0f;
};

struct VerifyStats {
    bool matched = false;
    bool faceDetected = false;
    bool isReal = false;
    float bestSimilarity = 0.0f;
    float realScore = 0.0f;
    double elapsedMs = 0.0;
    std::string error;
};

struct VaultInfo {
    bool hasEmbeddings = false;
    uint32_t enrolledCount = 0;
    float threshold = 0.45f;
    bool antiSpoofing = true;
    bool hasCredentials = false;
    std::wstring username;
    std::wstring domain;
};

// Zero-Daemon On-Demand C++20 Biometric Unlock Entrypoint (used directly by LogonUI.exe).
// Opens webcam via MediaFoundation + loads ONNX Runtime models in parallel, verifies
// live face + MiniFASNet anti-spoofing + SFace cosine match, shuts down webcam hardware
// immediately upon completion, and decrypts machine-vault DPAPI credentials.
bool NativeRequestUnlock(std::wstring& username,
                         std::wstring& password,
                         std::wstring& domain,
                         std::string& errorOut);

// Live webcam verification test without returning plaintext credentials (for Desktop test button).
// Reads at least maxFrames frames, and keeps reading until minScanMs has passed since the
// camera opened (first frames after power-on are often still dark), stopping early on a match.
VerifyStats NativeVerifyLiveCamera(int maxFrames = 6, int minScanMs = 0);

// Verify a single BGR24 image buffer (width x height, 3 bytes/pixel) against enrolled embeddings.
VerifyStats NativeVerifyBgrImage(const uint8_t* bgr, int width, int height, bool checkLiveness = true);

// Build C:\ProgramData\FaceUnlock\embeddings.bin from all .jpg/.png files in `dirPath` using GDI+ + ONNX.
bool NativeBuildEnrollmentFromDir(const std::wstring& dirPath,
                                  uint32_t& outCount,
                                  float threshold,
                                  bool antiSpoofing,
                                  std::string& errorOut);

// Capture `targetSamples` live frames from webcam (with callback for UI progress) and rebuild embeddings.bin.
using EnrollProgressCallback = void(*)(uint32_t captured, uint32_t target, const uint8_t* bgrFrame, int w, int h, const FaceBox5* face, void* userData);
bool NativeEnrollFromWebcam(uint32_t targetSamples,
                            float threshold,
                            bool antiSpoofing,
                            EnrollProgressCallback progressCb,
                            void* userData,
                            uint32_t& outCount,
                            std::string& errorOut);

// Machine-scope DPAPI credential vault management (requires Administrator or SYSTEM).
bool NativeSaveVaultCredentials(const std::wstring& domain,
                                const std::wstring& username,
                                const std::wstring& password,
                                std::string& errorOut);

bool NativeLoadVaultCredentials(std::wstring& domain,
                                std::wstring& username,
                                std::wstring& password,
                                std::string& errorOut);

VaultInfo NativeQueryVaultInfo();

}  // namespace FaceUnlock

// Exported C entrypoints for FaceIdManager.exe / diagnostics
extern "C" {
__declspec(dllexport) BOOL WINAPI NativeVerifyFaceLive(float* outSimilarity, BOOL* outIsReal, DWORD* outElapsedMs);
__declspec(dllexport) BOOL WINAPI NativeEnrollDirectoryW( LPCWSTR dirPath, DWORD* outCount);
}
