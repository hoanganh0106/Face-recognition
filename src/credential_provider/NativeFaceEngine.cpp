#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <wincrypt.h>
#include <objidl.h>
#include <gdiplus.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>

#include "NativeFaceEngine.h"
#include "onnxruntime_c_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace FaceUnlock {

namespace {

constexpr wchar_t kDataRoot[]      = L"C:\\ProgramData\\FaceUnlock";
constexpr wchar_t kOrtDllPath[]    = L"C:\\ProgramData\\FaceUnlock\\bin\\onnxruntime.dll";
constexpr wchar_t kModelYunet[]    = L"C:\\ProgramData\\FaceUnlock\\models\\yunet_opt.onnx";
constexpr wchar_t kModelSface[]    = L"C:\\ProgramData\\FaceUnlock\\models\\sface_opt.onnx";
constexpr wchar_t kModelFas1[]     = L"C:\\ProgramData\\FaceUnlock\\models\\fas1_opt.onnx";
constexpr wchar_t kModelFas2[]     = L"C:\\ProgramData\\FaceUnlock\\models\\fas2_opt.onnx";
constexpr wchar_t kEmbeddingsBin[] = L"C:\\ProgramData\\FaceUnlock\\embeddings.bin";
constexpr wchar_t kVaultFile[]     = L"C:\\ProgramData\\FaceUnlock\\vault\\credentials.dpapi";

constexpr char     kVaultEntropy[] = "FaceUnlockNativeCpp20Vault::v2.0";
constexpr uint32_t kEmbedMagic     = 0x46414345u;  // 'FACE'
constexpr uint32_t kEmbedVersion   = 1u;
constexpr uint32_t kEmbedDim       = 128u;

// Canonical 112x112 5-point landmarks for ArcFace / SFace alignment
constexpr float kCanonicalLandmarks[5][2] = {
    { 38.2946f, 51.6963f },
    { 73.5318f, 51.5014f },
    { 56.0252f, 71.7366f },
    { 41.5493f, 92.3655f },
    { 70.7299f, 92.2041f }
};

#pragma pack(push, 1)
struct EmbeddingsFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t dim;
    float    threshold;
    uint32_t antiSpoofing;
    uint32_t reserved[2];
};
#pragma pack(pop)

// Minimal RAII COM smart pointer
template <typename T>
class ComPtr {
public:
    ComPtr() noexcept : m_ptr(nullptr) {}
    explicit ComPtr(T* p) noexcept : m_ptr(p) {}
    ~ComPtr() { Reset(); }

    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;

    ComPtr(ComPtr&& other) noexcept : m_ptr(other.m_ptr) { other.m_ptr = nullptr; }
    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) {
            Reset();
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
        }
        return *this;
    }

    void Reset(T* p = nullptr) noexcept {
        if (m_ptr) m_ptr->Release();
        m_ptr = p;
    }

    T* Get() const noexcept { return m_ptr; }
    T* operator->() const noexcept { return m_ptr; }
    T** AddressOf() noexcept { Reset(); return &m_ptr; }
    explicit operator bool() const noexcept { return m_ptr != nullptr; }

private:
    T* m_ptr;
};

struct CoInitGuard {
    HRESULT hr;
    explicit CoInitGuard(DWORD coInit = COINIT_MULTITHREADED) noexcept {
        hr = CoInitializeEx(nullptr, coInit);
    }
    ~CoInitGuard() {
        if (SUCCEEDED(hr)) {
            CoUninitialize();
        }
    }
};

std::wstring Utf8ToWide(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

std::string WideToUtf8(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n, nullptr, nullptr);
    return out;
}

std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

void AppendUtf8Codepoint(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool ParseHex4(const std::string& s, size_t pos, uint32_t& val) {
    if (pos + 4 > s.size()) return false;
    val = 0;
    for (size_t i = 0; i < 4; ++i) {
        char c = s[pos + i];
        val <<= 4;
        if (c >= '0' && c <= '9') val |= static_cast<uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') val |= static_cast<uint32_t>(10 + (c - 'a'));
        else if (c >= 'A' && c <= 'F') val |= static_cast<uint32_t>(10 + (c - 'A'));
        else return false;
    }
    return true;
}

bool JsonExtractString(const std::string& json, const std::string& key, std::string& out) {
    std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    size_t colon = std::string::npos;
    while (true) {
        size_t k = json.find(needle, pos);
        if (k == std::string::npos) return false;
        size_t after = k + needle.size();
        while (after < json.size() && (json[after] == ' ' || json[after] == '\t' || json[after] == '\r' || json[after] == '\n')) {
            ++after;
        }
        if (after < json.size() && json[after] == ':') {
            colon = after;
            break;
        }
        pos = k + 1;
    }

    size_t q1 = colon + 1;
    while (q1 < json.size() && (json[q1] == ' ' || json[q1] == '\t' || json[q1] == '\r' || json[q1] == '\n')) ++q1;
    if (q1 >= json.size() || json[q1] != '"') return false;

    std::string decoded;
    size_t i = q1 + 1;
    while (i < json.size()) {
        char c = json[i++];
        if (c == '"') {
            out = std::move(decoded);
            return true;
        }
        if (c == '\\') {
            if (i >= json.size()) return false;
            char esc = json[i++];
            switch (esc) {
                case '"':  decoded.push_back('"');  break;
                case '\\': decoded.push_back('\\'); break;
                case '/':  decoded.push_back('/');  break;
                case 'b':  decoded.push_back('\b'); break;
                case 'f':  decoded.push_back('\f'); break;
                case 'n':  decoded.push_back('\n'); break;
                case 'r':  decoded.push_back('\r'); break;
                case 't':  decoded.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!ParseHex4(json, i, cp)) return false;
                    i += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= json.size() && json[i] == '\\' && json[i + 1] == 'u') {
                        uint32_t low = 0;
                        if (ParseHex4(json, i + 2, low) && low >= 0xDC00 && low <= 0xDFFF) {
                            i += 6;
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                        }
                    }
                    AppendUtf8Codepoint(decoded, cp);
                    break;
                }
                default:
                    decoded.push_back(esc);
                    break;
            }
        } else {
            decoded.push_back(c);
        }
    }
    return false;
}

// ============================================================================
// Bilinear Image Sampling & Closed-Form 5-Point Umeyama Similarity Warp
// ============================================================================

inline void SampleBilinearBgr(const uint8_t* bgr, int w, int h, float fx, float fy, float outBgr[3]) {
    if (fx < 0.0f) fx = 0.0f;
    if (fy < 0.0f) fy = 0.0f;
    if (fx > static_cast<float>(w - 1)) fx = static_cast<float>(w - 1);
    if (fy > static_cast<float>(h - 1)) fy = static_cast<float>(h - 1);

    int x0 = static_cast<int>(fx);
    int y0 = static_cast<int>(fy);
    int x1 = (x0 + 1 < w) ? (x0 + 1) : x0;
    int y1 = (y0 + 1 < h) ? (y0 + 1) : y0;

    float dx = fx - static_cast<float>(x0);
    float dy = fy - static_cast<float>(y0);
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    const uint8_t* p00 = bgr + (y0 * w + x0) * 3;
    const uint8_t* p10 = bgr + (y0 * w + x1) * 3;
    const uint8_t* p01 = bgr + (y1 * w + x0) * 3;
    const uint8_t* p11 = bgr + (y1 * w + x1) * 3;

    outBgr[0] = p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11;
    outBgr[1] = p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11;
    outBgr[2] = p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11;
}

// Warp face into 112x112 RGB NCHW float32 [0..255] tensor for SFace using exact 5-point similarity fit
void WarpAlignedFaceRgb112Nchw(const uint8_t* bgr,
                               int srcW,
                               int srcH,
                               const float srcPts[5][2],
                               float* dstNchwRgb) {
    // Fit inverse similarity mapping (u, v) in 112x112 -> (x, y) in source image:
    // x = a*u - b*v + tx
    // y = b*u + a*v + ty
    float meanU = 0.0f, meanV = 0.0f, meanX = 0.0f, meanY = 0.0f;
    for (int i = 0; i < 5; ++i) {
        meanU += kCanonicalLandmarks[i][0];
        meanV += kCanonicalLandmarks[i][1];
        meanX += srcPts[i][0];
        meanY += srcPts[i][1];
    }
    meanU *= 0.2f; meanV *= 0.2f;
    meanX *= 0.2f; meanY *= 0.2f;

    float denom = 0.0f, numA = 0.0f, numB = 0.0f;
    for (int i = 0; i < 5; ++i) {
        float du = kCanonicalLandmarks[i][0] - meanU;
        float dv = kCanonicalLandmarks[i][1] - meanV;
        float dx = srcPts[i][0] - meanX;
        float dy = srcPts[i][1] - meanY;
        denom += du * du + dv * dv;
        numA  += du * dx + dv * dy;
        numB  += du * dy - dv * dx;
    }

    float a = (denom > 1e-6f) ? (numA / denom) : 1.0f;
    float b = (denom > 1e-6f) ? (numB / denom) : 0.0f;
    float tx = meanX - (a * meanU - b * meanV);
    float ty = meanY - (b * meanU + a * meanV);

    constexpr int kOut = 112;
    constexpr int kPlane = kOut * kOut;
    float* chR = dstNchwRgb;
    float* chG = dstNchwRgb + kPlane;
    float* chB = dstNchwRgb + kPlane * 2;

    for (int v = 0; v < kOut; ++v) {
        float fv = static_cast<float>(v);
        for (int u = 0; u < kOut; ++u) {
            float fu = static_cast<float>(u);
            float sx = a * fu - b * fv + tx;
            float sy = b * fu + a * fv + ty;
            float pxBgr[3];
            SampleBilinearBgr(bgr, srcW, srcH, sx, sy, pxBgr);
            int idx = v * kOut + u;
            chR[idx] = pxBgr[2];  // Swap BGR -> RGB [0..255] matching cv2.FaceRecognizerSF
            chG[idx] = pxBgr[1];
            chB[idx] = pxBgr[0];
        }
    }
}

// Crop & resize to 80x80 BGR NHWC float32 [0..255] matching Minivision Silent-Face-Anti-Spoofing
void CropMiniFasNet80Nhwc(const uint8_t* bgr,
                          int srcW,
                          int srcH,
                          float bx,
                          float by,
                          float bw,
                          float bh,
                          float scale,
                          float* dstNhwcBgr) {
    float maxScaleH = (srcH - 1) / (std::max)(1.0f, bh);
    float maxScaleW = (srcW - 1) / (std::max)(1.0f, bw);
    float s = (std::min)({ maxScaleH, maxScaleW, scale });

    float newW = bw * s;
    float newH = bh * s;
    float cx = bx + bw * 0.5f;
    float cy = by + bh * 0.5f;

    float x1 = cx - newW * 0.5f;
    float y1 = cy - newH * 0.5f;
    float x2 = cx + newW * 0.5f;
    float y2 = cy + newH * 0.5f;

    if (x1 < 0.0f) { x2 -= x1; x1 = 0.0f; }
    if (y1 < 0.0f) { y2 -= y1; y1 = 0.0f; }
    if (x2 > static_cast<float>(srcW - 1)) {
        x1 -= (x2 - static_cast<float>(srcW - 1));
        x2 = static_cast<float>(srcW - 1);
    }
    if (y2 > static_cast<float>(srcH - 1)) {
        y1 -= (y2 - static_cast<float>(srcH - 1));
        y2 = static_cast<float>(srcH - 1);
    }

    int ix1 = (std::clamp)(static_cast<int>(x1), 0, srcW - 1);
    int iy1 = (std::clamp)(static_cast<int>(y1), 0, srcH - 1);
    int ix2 = (std::clamp)(static_cast<int>(x2), ix1, srcW - 1);
    int iy2 = (std::clamp)(static_cast<int>(y2), iy1, srcH - 1);

    float cropW = static_cast<float>(ix2 - ix1 + 1);
    float cropH = static_cast<float>(iy2 - iy1 + 1);

    constexpr int kOut = 80;
    for (int v = 0; v < kOut; ++v) {
        float sy = static_cast<float>(iy1) + (v + 0.5f) * (cropH / kOut) - 0.5f;
        for (int u = 0; u < kOut; ++u) {
            float sx = static_cast<float>(ix1) + (u + 0.5f) * (cropW / kOut) - 0.5f;
            float pxBgr[3];
            SampleBilinearBgr(bgr, srcW, srcH, sx, sy, pxBgr);
            float* out = dstNhwcBgr + (v * kOut + u) * 3;
            out[0] = pxBgr[0];
            out[1] = pxBgr[1];
            out[2] = pxBgr[2];
        }
    }
}

// ============================================================================
// RAII ONNX Runtime C-API Biometric Pipeline (Zero Static Link Dependency)
// ============================================================================

class OrtBiometricPipeline {
public:
    OrtBiometricPipeline() = default;
    ~OrtBiometricPipeline() { Shutdown(); }

    OrtBiometricPipeline(const OrtBiometricPipeline&) = delete;
    OrtBiometricPipeline& operator=(const OrtBiometricPipeline&) = delete;

    bool Initialize(bool loadAntiSpoofing, std::string& errorOut) {
        m_hDll = LoadLibraryExW(kOrtDllPath, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!m_hDll) {
            errorOut = "cannot-load-onnxruntime-dll";
            return false;
        }

        using GetApiBaseFn = const OrtApiBase* (ORT_API_CALL*)(void);
        auto getBase = reinterpret_cast<GetApiBaseFn>(GetProcAddress(m_hDll, "OrtGetApiBase"));
        if (!getBase || !getBase()) {
            errorOut = "invalid-onnxruntime-apibase";
            return false;
        }

        m_ort = getBase()->GetApi(ORT_API_VERSION);
        if (!m_ort) {
            m_ort = getBase()->GetApi(17);
        }
        if (!m_ort) {
            errorOut = "unsupported-onnxruntime-version";
            return false;
        }

        if (!CheckStatus(m_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "FaceUnlockCpp20", &m_env), errorOut)) {
            return false;
        }
        if (!CheckStatus(m_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m_memInfo), errorOut)) {
            return false;
        }

        // Load all ONNX sessions concurrently across CPU cores
        auto loadSess = [this](const wchar_t* path, int threads) -> std::pair<OrtSession*, std::string> {
            OrtSessionOptions* opts = nullptr;
            std::string err;
            if (!CheckStatus(m_ort->CreateSessionOptions(&opts), err)) {
                return { nullptr, err };
            }
            DiscardStatus(m_ort->SetIntraOpNumThreads(opts, threads));
            DiscardStatus(m_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_BASIC));
            OrtSession* sess = nullptr;
            bool ok = CheckStatus(m_ort->CreateSession(m_env, path, opts, &sess), err);
            m_ort->ReleaseSessionOptions(opts);
            return { ok ? sess : nullptr, err };
        };

        auto fDet = std::async(std::launch::async, loadSess, kModelYunet, 2);
        auto fRec = std::async(std::launch::async, loadSess, kModelSface, 4);
        std::future<std::pair<OrtSession*, std::string>> fFas1, fFas2;
        if (loadAntiSpoofing) {
            fFas1 = std::async(std::launch::async, loadSess, kModelFas1, 2);
            fFas2 = std::async(std::launch::async, loadSess, kModelFas2, 2);
        }

        auto rDet = fDet.get();
        auto rRec = fRec.get();
        m_sessYunet = rDet.first;
        m_sessSface = rRec.first;

        if (!m_sessYunet || !m_sessSface) {
            errorOut = !rDet.second.empty() ? rDet.second : rRec.second;
            return false;
        }

        if (loadAntiSpoofing) {
            auto rF1 = fFas1.get();
            auto rF2 = fFas2.get();
            m_sessFas1 = rF1.first;
            m_sessFas2 = rF2.first;
            if (!m_sessFas1 || !m_sessFas2) {
                errorOut = !rF1.second.empty() ? rF1.second : rF2.second;
                return false;
            }
        }
        return true;
    }

    void Shutdown() noexcept {
        if (m_ort) {
            if (m_sessFas2)  { m_ort->ReleaseSession(m_sessFas2);  m_sessFas2 = nullptr; }
            if (m_sessFas1)  { m_ort->ReleaseSession(m_sessFas1);  m_sessFas1 = nullptr; }
            if (m_sessSface) { m_ort->ReleaseSession(m_sessSface); m_sessSface = nullptr; }
            if (m_sessYunet) { m_ort->ReleaseSession(m_sessYunet); m_sessYunet = nullptr; }
            if (m_memInfo)   { m_ort->ReleaseMemoryInfo(m_memInfo); m_memInfo = nullptr; }
            if (m_env)       { m_ort->ReleaseEnv(m_env);           m_env = nullptr; }
            m_ort = nullptr;
        }
        if (m_hDll) {
            FreeLibrary(m_hDll);
            m_hDll = nullptr;
        }
    }

    // Detect largest prominent face in BGR24 frame using YuNet 640x640
    bool DetectFace(const uint8_t* bgr, int width, int height, FaceBox5& outFace, float scoreThresh = 0.65f) {
        if (!m_sessYunet || width <= 0 || height <= 0) return false;

        constexpr int kInSz = 640;
        float scale = 1.0f;
        if (width > kInSz || height > kInSz) {
            scale = (std::min)(static_cast<float>(kInSz) / width, static_cast<float>(kInSz) / height);
        }

        int scaledW = (std::min)(kInSz, static_cast<int>(std::round(width * scale)));
        int scaledH = (std::min)(kInSz, static_cast<int>(std::round(height * scale)));

        m_yunetBuf.assign(3 * kInSz * kInSz, 0.0f);
        float* chB = m_yunetBuf.data();
        float* chG = chB + kInSz * kInSz;
        float* chR = chG + kInSz * kInSz;

        if (scaledW == width && scaledH == height) {
            for (int y = 0; y < height; ++y) {
                const uint8_t* row = bgr + y * width * 3;
                int dstRow = y * kInSz;
                for (int x = 0; x < width; ++x) {
                    chB[dstRow + x] = static_cast<float>(row[x * 3 + 0]);
                    chG[dstRow + x] = static_cast<float>(row[x * 3 + 1]);
                    chR[dstRow + x] = static_cast<float>(row[x * 3 + 2]);
                }
            }
        } else {
            float invS = 1.0f / scale;
            for (int y = 0; y < scaledH; ++y) {
                float sy = (y + 0.5f) * invS - 0.5f;
                int dstRow = y * kInSz;
                for (int x = 0; x < scaledW; ++x) {
                    float sx = (x + 0.5f) * invS - 0.5f;
                    float px[3];
                    SampleBilinearBgr(bgr, width, height, sx, sy, px);
                    chB[dstRow + x] = px[0];
                    chG[dstRow + x] = px[1];
                    chR[dstRow + x] = px[2];
                }
            }
        }

        const int64_t inShape[4] = { 1, 3, kInSz, kInSz };
        OrtValue* inTensor = nullptr;
        std::string err;
        if (!CheckStatus(m_ort->CreateTensorWithDataAsOrtValue(
                m_memInfo, m_yunetBuf.data(), m_yunetBuf.size() * sizeof(float),
                inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inTensor), err)) {
            return false;
        }

        const char* inNames[1] = { "input" };
        const char* outNames[12] = {
            "cls_8", "cls_16", "cls_32",
            "obj_8", "obj_16", "obj_32",
            "bbox_8", "bbox_16", "bbox_32",
            "kps_8", "kps_16", "kps_32"
        };
        OrtValue* outTensors[12]{};

        bool runOk = CheckStatus(m_ort->Run(
            m_sessYunet, nullptr, inNames, &inTensor, 1, outNames, 12, outTensors), err);
        m_ort->ReleaseValue(inTensor);

        if (!runOk) {
            for (auto* v : outTensors) if (v) m_ort->ReleaseValue(v);
            return false;
        }

        constexpr int strides[3] = { 8, 16, 32 };
        constexpr int gridW[3]   = { 80, 40, 20 };
        constexpr int gridH[3]   = { 80, 40, 20 };

        std::vector<FaceBox5> candidates;
        candidates.reserve(32);
        float invScale = 1.0f / scale;

        for (int si = 0; si < 3; ++si) {
            float* clsData  = nullptr;
            float* objData  = nullptr;
            float* bboxData = nullptr;
            float* kpsData  = nullptr;
            DiscardStatus(m_ort->GetTensorMutableData(outTensors[0 + si], reinterpret_cast<void**>(&clsData)));
            DiscardStatus(m_ort->GetTensorMutableData(outTensors[3 + si], reinterpret_cast<void**>(&objData)));
            DiscardStatus(m_ort->GetTensorMutableData(outTensors[6 + si], reinterpret_cast<void**>(&bboxData)));
            DiscardStatus(m_ort->GetTensorMutableData(outTensors[9 + si], reinterpret_cast<void**>(&kpsData)));

            int s = strides[si];
            int gw = gridW[si];
            int gh = gridH[si];
            int cells = gw * gh;

            for (int idx = 0; idx < cells; ++idx) {
                float c = (std::clamp)(clsData[idx], 0.0f, 1.0f);
                float o = (std::clamp)(objData[idx], 0.0f, 1.0f);
                float score = std::sqrt(c * o);
                if (score < scoreThresh) continue;

                int r = idx / gw;
                int col = idx % gw;
                const float* bb = bboxData + idx * 4;
                const float* kp = kpsData  + idx * 10;

                float cx = (col + bb[0]) * s;
                float cy = (r   + bb[1]) * s;
                float bw = std::exp(bb[2]) * s;
                float bh = std::exp(bb[3]) * s;

                FaceBox5 fb{};
                fb.x = (cx - 0.5f * bw) * invScale;
                fb.y = (cy - 0.5f * bh) * invScale;
                fb.w = bw * invScale;
                fb.h = bh * invScale;
                fb.score = score;
                for (int k = 0; k < 5; ++k) {
                    fb.landmarks[k][0] = (col + kp[2 * k])     * s * invScale;
                    fb.landmarks[k][1] = (r   + kp[2 * k + 1]) * s * invScale;
                }
                candidates.push_back(fb);
            }
        }

        for (auto* v : outTensors) {
            if (v) m_ort->ReleaseValue(v);
        }

        if (candidates.empty()) return false;

        // Pick largest high-confidence face (foreground user in front of laptop)
        std::sort(candidates.begin(), candidates.end(), [](const FaceBox5& a, const FaceBox5& b) {
            return (a.w * a.h * a.score) > (b.w * b.h * b.score);
        });
        outFace = candidates.front();
        return true;
    }

    // Dual-branch MiniFASNetV2 + MiniFASNetV1SE anti-spoofing liveness check
    bool CheckLiveness(const uint8_t* bgr, int width, int height, const FaceBox5& face, float& outRealScore) {
        outRealScore = 1.0f;
        if (!m_sessFas1 || !m_sessFas2) return true;

        std::array<float, 80 * 80 * 3> buf1{};
        std::array<float, 80 * 80 * 3> buf2{};
        CropMiniFasNet80Nhwc(bgr, width, height, face.x, face.y, face.w, face.h, 2.7f, buf1.data());
        CropMiniFasNet80Nhwc(bgr, width, height, face.x, face.y, face.w, face.h, 4.0f, buf2.data());

        float probs1[3]{}, probs2[3]{};
        if (!RunFasSingle(m_sessFas1, buf1.data(), probs1) ||
            !RunFasSingle(m_sessFas2, buf2.data(), probs2)) {
            outRealScore = 0.0f;
            return false;
        }

        float sumProb[3] = {
            probs1[0] + probs2[0],
            probs1[1] + probs2[1],
            probs1[2] + probs2[2]
        };
        int bestIdx = 0;
        if (sumProb[1] > sumProb[bestIdx]) bestIdx = 1;
        if (sumProb[2] > sumProb[bestIdx]) bestIdx = 2;

        outRealScore = sumProb[1] * 0.5f;
        return bestIdx == 1;
    }

    // Compute L2-normalized 128-D SFace embedding
    bool ComputeEmbedding(const uint8_t* bgr, int width, int height, const FaceBox5& face, std::array<float, kEmbedDim>& outEmb) {
        if (!m_sessSface) return false;

        std::array<float, 3 * 112 * 112> alignedRgb{};
        WarpAlignedFaceRgb112Nchw(bgr, width, height, face.landmarks, alignedRgb.data());

        const int64_t inShape[4] = { 1, 3, 112, 112 };
        OrtValue* inTensor = nullptr;
        std::string err;
        if (!CheckStatus(m_ort->CreateTensorWithDataAsOrtValue(
                m_memInfo, alignedRgb.data(), alignedRgb.size() * sizeof(float),
                inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inTensor), err)) {
            return false;
        }

        const char* inNames[1]  = { "data" };
        const char* outNames[1] = { "fc1" };
        OrtValue* outTensor = nullptr;

        bool ok = CheckStatus(m_ort->Run(
            m_sessSface, nullptr, inNames, &inTensor, 1, outNames, 1, &outTensor), err);
        m_ort->ReleaseValue(inTensor);

        if (!ok || !outTensor) {
            if (outTensor) m_ort->ReleaseValue(outTensor);
            return false;
        }

        float* raw = nullptr;
        DiscardStatus(m_ort->GetTensorMutableData(outTensor, reinterpret_cast<void**>(&raw)));
        float normSq = 0.0f;
        for (size_t i = 0; i < kEmbedDim; ++i) {
            normSq += raw[i] * raw[i];
        }
        float invNorm = 1.0f / (std::sqrt(normSq) + 1e-9f);
        for (size_t i = 0; i < kEmbedDim; ++i) {
            outEmb[i] = raw[i] * invNorm;
        }
        m_ort->ReleaseValue(outTensor);
        return true;
    }

private:
    bool RunFasSingle(OrtSession* sess, float* nhwc80, float outSoftmax[3]) {
        const int64_t inShape[4] = { 1, 80, 80, 3 };
        OrtValue* inTensor = nullptr;
        std::string err;
        if (!CheckStatus(m_ort->CreateTensorWithDataAsOrtValue(
                m_memInfo, nhwc80, 80 * 80 * 3 * sizeof(float),
                inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inTensor), err)) {
            return false;
        }
        const char* inNames[1]  = { "input" };
        const char* outNames[1] = { "output" };
        OrtValue* outTensor = nullptr;
        bool ok = CheckStatus(m_ort->Run(
            sess, nullptr, inNames, &inTensor, 1, outNames, 1, &outTensor), err);
        m_ort->ReleaseValue(inTensor);
        if (!ok || !outTensor) {
            if (outTensor) m_ort->ReleaseValue(outTensor);
            return false;
        }

        float* logits = nullptr;
        DiscardStatus(m_ort->GetTensorMutableData(outTensor, reinterpret_cast<void**>(&logits)));
        float maxL = (std::max)({ logits[0], logits[1], logits[2] });
        float e0 = std::exp(logits[0] - maxL);
        float e1 = std::exp(logits[1] - maxL);
        float e2 = std::exp(logits[2] - maxL);
        float invSum = 1.0f / (e0 + e1 + e2 + 1e-9f);
        outSoftmax[0] = e0 * invSum;
        outSoftmax[1] = e1 * invSum;
        outSoftmax[2] = e2 * invSum;
        m_ort->ReleaseValue(outTensor);
        return true;
    }

    void DiscardStatus(OrtStatus* st) const noexcept {
        if (st && m_ort) {
            m_ort->ReleaseStatus(st);
        }
    }

    bool CheckStatus(OrtStatus* st, std::string& errorOut) const {
        if (!st) return true;
        const char* msg = m_ort->GetErrorMessage(st);
        errorOut = msg ? msg : "ort-error";
        m_ort->ReleaseStatus(st);
        return false;
    }

    HMODULE m_hDll = nullptr;
    const OrtApi* m_ort = nullptr;
    OrtEnv* m_env = nullptr;
    OrtMemoryInfo* m_memInfo = nullptr;
    OrtSession* m_sessYunet = nullptr;
    OrtSession* m_sessSface = nullptr;
    OrtSession* m_sessFas1  = nullptr;
    OrtSession* m_sessFas2  = nullptr;
    std::vector<float> m_yunetBuf;
};

// ============================================================================
// RAII Win32 MediaFoundation Webcam (Zero External Camera Dependencies)
// ============================================================================

class MfWebcam {
public:
    MfWebcam() = default;
    ~MfWebcam() { Close(); }

    MfWebcam(const MfWebcam&) = delete;
    MfWebcam& operator=(const MfWebcam&) = delete;

    bool Open(uint32_t deviceIndex, std::string& errorOut) {
        Close();
        HRESULT hr = MFStartup(MF_VERSION);
        if (FAILED(hr)) {
            errorOut = "mfstartup-failed";
            return false;
        }
        m_mfStarted = true;

        ComPtr<IMFAttributes> enumAttr;
        if (FAILED(MFCreateAttributes(enumAttr.AddressOf(), 1)) ||
            FAILED(enumAttr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                     MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID))) {
            errorOut = "mf-enum-attr-failed";
            Close();
            return false;
        }

        IMFActivate** devices = nullptr;
        UINT32 count = 0;
        hr = MFEnumDeviceSources(enumAttr.Get(), &devices, &count);
        if (FAILED(hr) || count == 0 || !devices) {
            errorOut = "no-webcam-found";
            Close();
            return false;
        }

        uint32_t chosen = (deviceIndex < count) ? deviceIndex : 0;
        hr = devices[chosen]->ActivateObject(__uuidof(IMFMediaSource),
                                             reinterpret_cast<void**>(m_source.AddressOf()));
        for (UINT32 i = 0; i < count; ++i) {
            devices[i]->Release();
        }
        CoTaskMemFree(devices);

        if (FAILED(hr) || !m_source) {
            errorOut = "webcam-activate-failed";
            Close();
            return false;
        }

        ComPtr<IMFAttributes> readerAttr;
        if (FAILED(MFCreateAttributes(readerAttr.AddressOf(), 2))) {
            errorOut = "mf-reader-attr-failed";
            Close();
            return false;
        }
        // Enable MediaFoundation hardware/SIMD YUV/MJPG -> RGB32 conversion
        readerAttr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
        readerAttr->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, FALSE);

        hr = MFCreateSourceReaderFromMediaSource(m_source.Get(), readerAttr.Get(), m_reader.AddressOf());
        if (FAILED(hr) || !m_reader) {
            errorOut = "mf-create-reader-failed";
            Close();
            return false;
        }

        // Request RGB32 output (try 640x480 first, then camera's native resolution in RGB32)
        ComPtr<IMFMediaType> outType;
        if (SUCCEEDED(MFCreateMediaType(outType.AddressOf()))) {
            outType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
            outType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
            MFSetAttributeSize(outType.Get(), MF_MT_FRAME_SIZE, 640, 480);
            hr = m_reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, outType.Get());
            if (FAILED(hr)) {
                ComPtr<IMFMediaType> fallbackType;
                if (SUCCEEDED(MFCreateMediaType(fallbackType.AddressOf()))) {
                    fallbackType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
                    fallbackType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
                    hr = m_reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, fallbackType.Get());
                }
            }
        }
        if (FAILED(hr)) {
            errorOut = "mf-set-rgb32-failed";
            Close();
            return false;
        }

        // Query negotiated width, height, and signed default stride
        ComPtr<IMFMediaType> curType;
        if (FAILED(m_reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, curType.AddressOf()))) {
            errorOut = "mf-get-current-type-failed";
            Close();
            return false;
        }
        UINT32 w = 0, h = 0;
        MFGetAttributeSize(curType.Get(), MF_MT_FRAME_SIZE, &w, &h);
        m_width  = static_cast<int>(w);
        m_height = static_cast<int>(h);

        UINT32 rawStride = 0;
        if (SUCCEEDED(curType->GetUINT32(MF_MT_DEFAULT_STRIDE, &rawStride))) {
            m_defaultStride = static_cast<LONG>(static_cast<INT32>(rawStride));
        } else {
            m_defaultStride = m_width * 4;
        }

        if (m_width <= 0 || m_height <= 0) {
            errorOut = "invalid-camera-dimensions";
            Close();
            return false;
        }

        // Drain 2 initial frames so UVC auto-exposure settles
        std::vector<uint8_t> dummy;
        int dw = 0, dh = 0;
        for (int i = 0; i < 2; ++i) {
            ReadFrameBgr(dummy, dw, dh);
        }
        return true;
    }

    bool ReadFrameBgr(std::vector<uint8_t>& outBgr, int& outW, int& outH) {
        if (!m_reader) return false;

        for (int tries = 0; tries < 8; ++tries) {
            DWORD streamIndex = 0, flags = 0;
            LONGLONG timestamp = 0;
            ComPtr<IMFSample> sample;

            HRESULT hr = m_reader->ReadSample(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                0,
                &streamIndex,
                &flags,
                &timestamp,
                sample.AddressOf());

            if (FAILED(hr) || (flags & MF_SOURCE_READERF_ERROR)) {
                return false;
            }
            if (!sample) {
                Sleep(15);
                continue;
            }

            ComPtr<IMFMediaBuffer> mediaBuf;
            if (FAILED(sample->ConvertToContiguousBuffer(mediaBuf.AddressOf())) || !mediaBuf) {
                continue;
            }

            outW = m_width;
            outH = m_height;
            outBgr.resize(static_cast<size_t>(m_width) * static_cast<size_t>(m_height) * 3u);

            // Prefer IMF2DBuffer::Lock2D so scanline0 is guaranteed top row regardless of DIB orientation
            ComPtr<IMF2DBuffer> buf2D;
            if (SUCCEEDED(mediaBuf->QueryInterface(__uuidof(IMF2DBuffer),
                                                   reinterpret_cast<void**>(buf2D.AddressOf()))) && buf2D) {
                BYTE* scanline0 = nullptr;
                LONG pitch = 0;
                if (SUCCEEDED(buf2D->Lock2D(&scanline0, &pitch))) {
                    CopyBgxToBgr(scanline0, pitch, m_width, m_height, outBgr.data());
                    buf2D->Unlock2D();
                    return true;
                }
            }

            BYTE* rawPtr = nullptr;
            DWORD maxLen = 0, curLen = 0;
            if (SUCCEEDED(mediaBuf->Lock(&rawPtr, &maxLen, &curLen))) {
                LONG pitch = (m_defaultStride != 0) ? m_defaultStride : (m_width * 4);
                BYTE* scanline0 = (pitch >= 0)
                    ? rawPtr
                    : (rawPtr + static_cast<ptrdiff_t>(m_height - 1) * (-pitch));
                CopyBgxToBgr(scanline0, pitch, m_width, m_height, outBgr.data());
                mediaBuf->Unlock();
                return true;
            }
        }
        return false;
    }

    // Immediate hardware shutdown — turns webcam LED OFF right away
    void Close() noexcept {
        m_reader.Reset();
        if (m_source) {
            m_source->Shutdown();
            m_source.Reset();
        }
        if (m_mfStarted) {
            MFShutdown();
            m_mfStarted = false;
        }
    }

private:
    static void CopyBgxToBgr(const BYTE* scanline0, LONG pitch, int w, int h, uint8_t* dstBgr) noexcept {
        for (int y = 0; y < h; ++y) {
            const BYTE* srcRow = scanline0 + static_cast<ptrdiff_t>(y) * pitch;
            uint8_t* dstRow = dstBgr + static_cast<size_t>(y) * static_cast<size_t>(w) * 3u;
            for (int x = 0; x < w; ++x) {
                dstRow[x * 3 + 0] = srcRow[x * 4 + 0];  // B
                dstRow[x * 3 + 1] = srcRow[x * 4 + 1];  // G
                dstRow[x * 3 + 2] = srcRow[x * 4 + 2];  // R
            }
        }
    }

    ComPtr<IMFMediaSource> m_source;
    ComPtr<IMFSourceReader> m_reader;
    int m_width = 0;
    int m_height = 0;
    LONG m_defaultStride = 0;
    bool m_mfStarted = false;
};

// ============================================================================
// Enrolled Embeddings Storage
// ============================================================================

struct EnrolledDatabase {
    float threshold = 0.45f;
    bool antiSpoofing = true;
    std::vector<std::array<float, kEmbedDim>> refs;

    bool Load(std::string& errorOut) {
        std::ifstream f(std::filesystem::path(kEmbeddingsBin), std::ios::binary);
        if (!f) {
            errorOut = "no-embeddings-file";
            return false;
        }
        EmbeddingsFileHeader hdr{};
        if (!f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) {
            errorOut = "corrupt-embeddings-header";
            return false;
        }
        if (hdr.magic != kEmbedMagic || hdr.dim != kEmbedDim || hdr.count == 0 || hdr.count > 500) {
            errorOut = "invalid-embeddings-header";
            return false;
        }
        threshold    = hdr.threshold;
        antiSpoofing = (hdr.antiSpoofing != 0);
        refs.resize(hdr.count);
        for (uint32_t i = 0; i < hdr.count; ++i) {
            if (!f.read(reinterpret_cast<char*>(refs[i].data()), kEmbedDim * sizeof(float))) {
                errorOut = "truncated-embeddings-file";
                return false;
            }
        }
        return true;
    }

    bool Save(std::string& errorOut) const {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(kDataRoot), ec);
        std::ofstream f(std::filesystem::path(kEmbeddingsBin), std::ios::binary | std::ios::trunc);
        if (!f) {
            errorOut = "cannot-write-embeddings-file";
            return false;
        }
        EmbeddingsFileHeader hdr{};
        hdr.magic        = kEmbedMagic;
        hdr.version      = kEmbedVersion;
        hdr.count        = static_cast<uint32_t>(refs.size());
        hdr.dim          = kEmbedDim;
        hdr.threshold    = threshold;
        hdr.antiSpoofing = antiSpoofing ? 1u : 0u;
        f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        for (const auto& vec : refs) {
            f.write(reinterpret_cast<const char*>(vec.data()), kEmbedDim * sizeof(float));
        }
        return f.good();
    }

    float BestCosineSimilarity(const std::array<float, kEmbedDim>& probe) const noexcept {
        float best = -1.0f;
        for (const auto& r : refs) {
            float dot = 0.0f;
            for (size_t i = 0; i < kEmbedDim; ++i) {
                dot += probe[i] * r[i];
            }
            if (dot > best) best = dot;
        }
        return best;
    }
};

// Decode a .jpg/.png image into BGR24 using native Win32 GDI+
bool LoadImageFileBgr(const std::wstring& filePath, std::vector<uint8_t>& outBgr, int& outW, int& outH) {
    Gdiplus::Bitmap bmp(filePath.c_str());
    if (bmp.GetLastStatus() != Gdiplus::Ok) return false;
    UINT w = bmp.GetWidth();
    UINT h = bmp.GetHeight();
    if (w == 0 || h == 0) return false;

    Gdiplus::Rect rect(0, 0, static_cast<INT>(w), static_cast<INT>(h));
    Gdiplus::BitmapData bd{};
    if (bmp.LockBits(&rect, Gdiplus::ImageLockModeRead, PixelFormat24bppRGB, &bd) != Gdiplus::Ok) {
        return false;
    }

    outW = static_cast<int>(w);
    outH = static_cast<int>(h);
    outBgr.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);

    const uint8_t* srcBase = static_cast<const uint8_t*>(bd.Scan0);
    for (UINT y = 0; y < h; ++y) {
        const uint8_t* srcRow = srcBase + static_cast<ptrdiff_t>(y) * bd.Stride;
        std::memcpy(outBgr.data() + static_cast<size_t>(y) * w * 3u, srcRow, static_cast<size_t>(w) * 3u);
    }
    bmp.UnlockBits(&bd);
    return true;
}

}  // namespace

// ============================================================================
// Public Engine API Implementations
// ============================================================================

bool NativeSaveVaultCredentials(const std::wstring& domain,
                                const std::wstring& username,
                                const std::wstring& password,
                                std::string& errorOut) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(kVaultFile).parent_path(), ec);

    std::string u8 = WideToUtf8(username);
    std::string p8 = WideToUtf8(password);
    std::string d8 = WideToUtf8(domain);

    std::string json = "{\"u\":\"" + JsonEscape(u8) +
                       "\",\"p\":\"" + JsonEscape(p8) +
                       "\",\"d\":\"" + JsonEscape(d8) + "\"}";

    DATA_BLOB inBlob{};
    inBlob.pbData = reinterpret_cast<BYTE*>(json.data());
    inBlob.cbData = static_cast<DWORD>(json.size());

    DATA_BLOB entropyBlob{};
    entropyBlob.pbData = reinterpret_cast<BYTE*>(const_cast<char*>(kVaultEntropy));
    entropyBlob.cbData = static_cast<DWORD>(sizeof(kVaultEntropy) - 1);

    DATA_BLOB outBlob{};
    BOOL ok = CryptProtectData(
        &inBlob,
        L"FaceUnlockMachineVault",
        &entropyBlob,
        nullptr,
        nullptr,
        CRYPTPROTECT_LOCAL_MACHINE | CRYPTPROTECT_UI_FORBIDDEN,
        &outBlob);

    SecureZeroMemory(p8.data(), p8.size());
    SecureZeroMemory(json.data(), json.size());

    if (!ok) {
        errorOut = "dpapi-encrypt-failed";
        return false;
    }

    std::ofstream f(std::filesystem::path(kVaultFile), std::ios::binary | std::ios::trunc);
    bool written = false;
    if (f) {
        f.write(reinterpret_cast<const char*>(outBlob.pbData), outBlob.cbData);
        written = f.good();
    }
    LocalFree(outBlob.pbData);

    if (!written) {
        errorOut = "vault-write-denied";
        return false;
    }
    return true;
}

bool NativeLoadVaultCredentials(std::wstring& domain,
                                std::wstring& username,
                                std::wstring& password,
                                std::string& errorOut) {
    std::ifstream f(std::filesystem::path(kVaultFile), std::ios::binary | std::ios::ate);
    if (!f) {
        errorOut = "no-vault-credentials";
        return false;
    }
    std::streamsize sz = f.tellg();
    if (sz <= 0 || sz > 65536) {
        errorOut = "invalid-vault-size";
        return false;
    }
    f.seekg(0, std::ios::beg);
    std::vector<BYTE> enc(static_cast<size_t>(sz));
    if (!f.read(reinterpret_cast<char*>(enc.data()), sz)) {
        errorOut = "vault-read-failed";
        return false;
    }

    DATA_BLOB inBlob{ static_cast<DWORD>(enc.size()), enc.data() };
    DATA_BLOB entropyBlob{
        static_cast<DWORD>(sizeof(kVaultEntropy) - 1),
        reinterpret_cast<BYTE*>(const_cast<char*>(kVaultEntropy))
    };
    DATA_BLOB outBlob{};

    if (!CryptUnprotectData(&inBlob, nullptr, &entropyBlob, nullptr, nullptr,
                            CRYPTPROTECT_UI_FORBIDDEN, &outBlob)) {
        errorOut = "dpapi-decrypt-failed";
        return false;
    }

    std::string json(reinterpret_cast<const char*>(outBlob.pbData), outBlob.cbData);
    SecureZeroMemory(outBlob.pbData, outBlob.cbData);
    LocalFree(outBlob.pbData);

    std::string u8, p8, d8;
    bool parseOk = JsonExtractString(json, "u", u8) && JsonExtractString(json, "p", p8);
    if (!JsonExtractString(json, "d", d8)) {
        d8 = ".";
    }
    SecureZeroMemory(json.data(), json.size());

    if (!parseOk) {
        SecureZeroMemory(p8.data(), p8.size());
        errorOut = "corrupt-vault-json";
        return false;
    }

    username = Utf8ToWide(u8);
    password = Utf8ToWide(p8);
    domain   = Utf8ToWide(d8);
    SecureZeroMemory(p8.data(), p8.size());
    return true;
}

VaultInfo NativeQueryVaultInfo() {
    VaultInfo info{};
    EnrolledDatabase db;
    std::string err;
    if (db.Load(err)) {
        info.hasEmbeddings = true;
        info.enrolledCount = static_cast<uint32_t>(db.refs.size());
        info.threshold     = db.threshold;
        info.antiSpoofing  = db.antiSpoofing;
    }
    std::wstring pw;
    if (NativeLoadVaultCredentials(info.domain, info.username, pw, err)) {
        info.hasCredentials = true;
        SecureZeroMemory(pw.data(), pw.size() * sizeof(wchar_t));
    }
    return info;
}

VerifyStats NativeVerifyLiveCamera(int maxFrames) {
    auto t0 = std::chrono::steady_clock::now();
    VerifyStats stats{};

    EnrolledDatabase db;
    if (!db.Load(stats.error)) {
        return stats;
    }

    // Launch Webcam hardware initialization and ONNX Runtime session loading IN PARALLEL
    auto camFuture = std::async(std::launch::async, []() -> std::pair<std::unique_ptr<MfWebcam>, std::string> {
        CoInitGuard com(COINIT_MULTITHREADED);
        auto cam = std::make_unique<MfWebcam>();
        std::string err;
        if (!cam->Open(0, err)) {
            return { nullptr, err };
        }
        return { std::move(cam), "" };
    });

    OrtBiometricPipeline ortPipe;
    if (!ortPipe.Initialize(db.antiSpoofing, stats.error)) {
        auto camRes = camFuture.get();
        if (camRes.first) camRes.first->Close();
        return stats;
    }

    CoInitGuard comMain(COINIT_MULTITHREADED);
    auto camRes = camFuture.get();
    if (!camRes.first) {
        stats.error = camRes.second;
        return stats;
    }

    std::unique_ptr<MfWebcam> cam = std::move(camRes.first);
    std::vector<uint8_t> bgrFrame;
    int fw = 0, fh = 0;

    int totalFrames = (std::max)(1, maxFrames);
    for (int i = 0; i < totalFrames; ++i) {
        if (!cam->ReadFrameBgr(bgrFrame, fw, fh)) {
            continue;
        }

        FaceBox5 face{};
        if (!ortPipe.DetectFace(bgrFrame.data(), fw, fh, face)) {
            continue;
        }
        stats.faceDetected = true;

        float realScore = 1.0f;
        bool real = ortPipe.CheckLiveness(bgrFrame.data(), fw, fh, face, realScore);
        if (realScore > stats.realScore) stats.realScore = realScore;
        if (!real) {
            continue;
        }
        stats.isReal = true;

        std::array<float, kEmbedDim> emb{};
        if (!ortPipe.ComputeEmbedding(bgrFrame.data(), fw, fh, face, emb)) {
            continue;
        }

        float sim = db.BestCosineSimilarity(emb);
        if (sim > stats.bestSimilarity) {
            stats.bestSimilarity = sim;
        }
        if (sim >= db.threshold) {
            stats.matched = true;
            break;  // Instant match — shut down camera hardware immediately!
        }
    }

    // Explicitly shut down hardware webcam BEFORE returning so LED turns OFF immediately
    cam->Close();
    cam.reset();

    stats.elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (!stats.matched && stats.error.empty()) {
        stats.error = stats.faceDetected ? (stats.isReal ? "face-mismatch" : "spoof-detected") : "no-face";
    }
    return stats;
}

bool NativeRequestUnlock(std::wstring& username,
                         std::wstring& password,
                         std::wstring& domain,
                         std::string& errorOut) {
    VerifyStats vs = NativeVerifyLiveCamera(6);
    if (!vs.matched) {
        errorOut = vs.error.empty() ? "no-match" : vs.error;
        return false;
    }
    return NativeLoadVaultCredentials(domain, username, password, errorOut);
}

VerifyStats NativeVerifyBgrImage(const uint8_t* bgr, int width, int height, bool checkLiveness) {
    auto t0 = std::chrono::steady_clock::now();
    VerifyStats stats{};

    EnrolledDatabase db;
    if (!db.Load(stats.error)) return stats;

    OrtBiometricPipeline ortPipe;
    if (!ortPipe.Initialize(checkLiveness && db.antiSpoofing, stats.error)) return stats;

    FaceBox5 face{};
    if (!ortPipe.DetectFace(bgr, width, height, face)) {
        stats.error = "no-face";
        return stats;
    }
    stats.faceDetected = true;

    if (checkLiveness && db.antiSpoofing) {
        stats.isReal = ortPipe.CheckLiveness(bgr, width, height, face, stats.realScore);
        if (!stats.isReal) {
            stats.error = "spoof-detected";
            return stats;
        }
    } else {
        stats.isReal = true;
        stats.realScore = 1.0f;
    }

    std::array<float, kEmbedDim> emb{};
    if (!ortPipe.ComputeEmbedding(bgr, width, height, face, emb)) {
        stats.error = "embed-failed";
        return stats;
    }

    stats.bestSimilarity = db.BestCosineSimilarity(emb);
    stats.matched = (stats.bestSimilarity >= db.threshold);
    stats.elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return stats;
}

bool NativeBuildEnrollmentFromDir(const std::wstring& dirPath,
                                  uint32_t& outCount,
                                  float threshold,
                                  bool antiSpoofing,
                                  std::string& errorOut) {
    outCount = 0;
    Gdiplus::GdiplusStartupInput gdipIn;
    ULONG_PTR gdipTok = 0;
    if (Gdiplus::GdiplusStartup(&gdipTok, &gdipIn, nullptr) != Gdiplus::Ok) {
        errorOut = "gdiplus-init-failed";
        return false;
    }

    bool result = false;
    {
        OrtBiometricPipeline ortPipe;
        if (!ortPipe.Initialize(false, errorOut)) {
            Gdiplus::GdiplusShutdown(gdipTok);
            return false;
        }

        EnrolledDatabase db;
        db.threshold    = threshold;
        db.antiSpoofing = antiSpoofing;

        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(dirPath), ec)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().wstring();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
            if (ext != L".jpg" && ext != L".jpeg" && ext != L".png" && ext != L".bmp") continue;

            std::vector<uint8_t> bgr;
            int w = 0, h = 0;
            if (!LoadImageFileBgr(entry.path().wstring(), bgr, w, h)) continue;

            FaceBox5 face{};
            if (!ortPipe.DetectFace(bgr.data(), w, h, face, 0.60f)) continue;

            std::array<float, kEmbedDim> emb{};
            if (ortPipe.ComputeEmbedding(bgr.data(), w, h, face, emb)) {
                db.refs.push_back(emb);
            }
        }

        if (db.refs.empty()) {
            errorOut = "no-valid-faces-found";
        } else if (db.Save(errorOut)) {
            outCount = static_cast<uint32_t>(db.refs.size());
            result = true;
        }
    }

    Gdiplus::GdiplusShutdown(gdipTok);
    return result;
}

bool NativeEnrollFromWebcam(uint32_t targetSamples,
                            float threshold,
                            bool antiSpoofing,
                            EnrollProgressCallback progressCb,
                            void* userData,
                            uint32_t& outCount,
                            std::string& errorOut) {
    outCount = 0;
    CoInitGuard com(COINIT_MULTITHREADED);

    OrtBiometricPipeline ortPipe;
    if (!ortPipe.Initialize(false, errorOut)) return false;

    MfWebcam cam;
    if (!cam.Open(0, errorOut)) return false;

    EnrolledDatabase db;
    db.threshold    = threshold;
    db.antiSpoofing = antiSpoofing;
    db.refs.reserve(targetSamples);

    std::vector<uint8_t> bgr;
    int w = 0, h = 0;
    ULONGLONG lastCaptureTick = 0;
    ULONGLONG deadline = GetTickCount64() + 45000ULL;

    while (db.refs.size() < targetSamples && GetTickCount64() < deadline) {
        if (!cam.ReadFrameBgr(bgr, w, h)) {
            Sleep(20);
            continue;
        }
        FaceBox5 face{};
        bool hasFace = ortPipe.DetectFace(bgr.data(), w, h, face, 0.68f);

        ULONGLONG now = GetTickCount64();
        if (hasFace && (now - lastCaptureTick >= 450ULL)) {
            std::array<float, kEmbedDim> emb{};
            if (ortPipe.ComputeEmbedding(bgr.data(), w, h, face, emb)) {
                db.refs.push_back(emb);
                lastCaptureTick = now;
            }
        }

        if (progressCb) {
            progressCb(static_cast<uint32_t>(db.refs.size()), targetSamples,
                       bgr.data(), w, h, hasFace ? &face : nullptr, userData);
        }
        Sleep(25);
    }

    cam.Close();

    if (db.refs.empty()) {
        errorOut = "no-faces-captured";
        return false;
    }
    if (!db.Save(errorOut)) {
        return false;
    }
    outCount = static_cast<uint32_t>(db.refs.size());
    return true;
}

}  // namespace FaceUnlock

extern "C" {

__declspec(dllexport) BOOL WINAPI NativeVerifyFaceLive(float* outSimilarity, BOOL* outIsReal, DWORD* outElapsedMs) {
    auto stats = FaceUnlock::NativeVerifyLiveCamera(6);
    if (outSimilarity) *outSimilarity = stats.bestSimilarity;
    if (outIsReal)     *outIsReal     = stats.isReal ? TRUE : FALSE;
    if (outElapsedMs)  *outElapsedMs  = static_cast<DWORD>(std::round(stats.elapsedMs));
    return stats.matched ? TRUE : FALSE;
}

__declspec(dllexport) BOOL WINAPI NativeEnrollDirectoryW(LPCWSTR dirPath, DWORD* outCount) {
    if (!dirPath) return FALSE;
    uint32_t cnt = 0;
    std::string err;
    bool ok = FaceUnlock::NativeBuildEnrollmentFromDir(dirPath, cnt, 0.45f, true, err);
    if (outCount) *outCount = cnt;
    return ok ? TRUE : FALSE;
}

}
