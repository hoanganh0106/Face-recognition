#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CpDir    = Join-Path $RepoRoot 'src\credential_provider'
$MgrCpp   = Join-Path $RepoRoot 'src\manager\FaceIdManager.cpp'
$IncDir   = Join-Path $RepoRoot 'third_party'
$DistDir  = Join-Path $RepoRoot 'dist'

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

$ClangCmd = Get-Command clang++ -ErrorAction SilentlyContinue
if ($ClangCmd) {
    $clang = $ClangCmd.Source
    Write-Host "[1/2] Building dist\FaceCredentialProvider.dll via clang++ (C++20 -O3 -mavx2 -mfma)..." -ForegroundColor Cyan
    Push-Location $CpDir
    try {
        & $clang -std=c++20 -O3 -mavx2 -mfma -Wall -Wextra -shared `
            -DUNICODE -D_UNICODE -DWIN32_LEAN_AND_MEAN -I"$IncDir" `
            dll.cpp ClassFactory.cpp FaceCredentialProvider.cpp FaceCredential.cpp FaceIdHud.cpp NativeFaceEngine.cpp helpers.cpp FaceCredentialProvider.def `
            -lcrypt32 -lmfplat -lmf -lmfreadwrite -lmfuuid -lcredui -lsecur32 -lshlwapi -lole32 -loleaut32 -ladvapi32 -luser32 -lgdi32 -lgdiplus -luuid `
            -static-libgcc -static-libstdc++ "-Wl,--kill-at" `
            -o "$DistDir\FaceCredentialProvider.dll"
        if ($LASTEXITCODE -ne 0) { throw "Failed to compile FaceCredentialProvider.dll" }

        Write-Host "[2/2] Building FaceIdManager.exe via clang++ (C++20 -O3 -mavx2 -mfma)..." -ForegroundColor Cyan
        & $clang -std=c++20 -O3 -mavx2 -mfma -Wall -Wextra -municode -mwindows `
            -DUNICODE -D_UNICODE -DWIN32_LEAN_AND_MEAN -I"$IncDir" -I"$CpDir" `
            "$MgrCpp" FaceIdHud.cpp NativeFaceEngine.cpp `
            -lcomctl32 -lshell32 -lcrypt32 -lmfplat -lmf -lmfreadwrite -lmfuuid -lole32 -loleaut32 -ladvapi32 -luser32 -lgdi32 -lgdiplus -luuid `
            -static-libgcc -static-libstdc++ `
            -o "$RepoRoot\FaceIdManager.exe"
        if ($LASTEXITCODE -ne 0) { throw "Failed to compile FaceIdManager.exe" }
    } finally {
        Pop-Location
    }
} else {
    $CmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $CmakeCmd) {
        throw "Neither clang++ nor cmake was found in PATH. Install LLVM-MinGW or Visual Studio C++ Build Tools."
    }
    $BuildDir = Join-Path $RepoRoot 'build_cmake'
    & cmake -S $RepoRoot -B $BuildDir -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
    & cmake --build $BuildDir --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }

    $BuiltDll = Get-ChildItem -Path $BuildDir -Recurse -Filter 'FaceCredentialProvider.dll' | Select-Object -First 1
    $BuiltExe = Get-ChildItem -Path $BuildDir -Recurse -Filter 'FaceIdManager.exe' | Select-Object -First 1
    if ($BuiltDll) { Copy-Item $BuiltDll.FullName "$DistDir\FaceCredentialProvider.dll" -Force }
    if ($BuiltExe) { Copy-Item $BuiltExe.FullName "$RepoRoot\FaceIdManager.exe" -Force }
}

Write-Host "`nBuild completed successfully!" -ForegroundColor Green
Get-Item "$DistDir\FaceCredentialProvider.dll", "$RepoRoot\FaceIdManager.exe" | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
